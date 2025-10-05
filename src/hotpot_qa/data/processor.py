"""
Long Context HotpotQA Dataset Processor
Creates datasets with extended contexts by adding distractor articles.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

import tiktoken

from hotpot_qa.utils.json_stream import JSONStreamer
from hotpot_qa.utils.tokenization import TokenCounter

logger = logging.getLogger(__name__)

@dataclass
class ContextConfig:
    """Configuration for context sizes."""
    size_name: str
    target_tokens: int
    tolerance_percent: float = 5.0

    @property
    def min_tokens(self) -> int:
        return int(self.target_tokens * (1 - self.tolerance_percent / 100))

    @property
    def max_tokens(self) -> int:
        return int(self.target_tokens * (1 + self.tolerance_percent / 100))

class LongContextProcessor:
    """Processes HotpotQA questions into long-context datasets."""

    def __init__(self,
                 hotpot_file: str = "hotpot_dev_fullwiki_v1.json",
                 catalog_file: str = "data/article_catalog.json",
                 gold_file: str = "data/gold_mappings.json",
                 distractor_file: str = "data/distractor_pools.json"):

        self.hotpot_file = Path(hotpot_file)
        self.catalog_file = Path(catalog_file)
        self.gold_file = Path(gold_file)
        self.distractor_file = Path(distractor_file)

        # Initialize utilities
        self.token_counter = TokenCounter()

        # Context configurations
        self.context_configs = {
            "8k": ContextConfig("8k", 8000),
            "32k": ContextConfig("32k", 32000),
            "128k": ContextConfig("128k", 128000)
        }

        # Cache for loaded resources
        self._catalog_cache: Optional[Dict] = None
        self._distractor_streamer: Optional[JSONStreamer] = None

        # Random seed for reproducibility
        random.seed(42)

    def load_hotpot_data(self) -> List[Dict[str, Any]]:
        """Load HotpotQA dataset."""
        logger.info(f"Loading HotpotQA data from {self.hotpot_file}")
        with open(self.hotpot_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} questions")
        return data

    def get_gold_articles(self, question: Dict[str, Any]) -> List[str]:
        """Extract gold article titles from supporting facts."""
        gold_titles = []
        for fact in question["supporting_facts"]:
            gold_titles.append(fact[0])
        return gold_titles

    def _load_catalog(self) -> Dict[str, Any]:
        """Load article catalog (cached)."""
        if self._catalog_cache is None:
            catalog_streamer = JSONStreamer(str(self.catalog_file))
            self._catalog_cache = catalog_streamer.load_small_file()
            logger.info(f"Loaded catalog with {len(self._catalog_cache)} articles")
        return self._catalog_cache

    def extract_gold_docs(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract gold documents with full text from context."""
        gold_titles = self.get_gold_articles(question)
        gold_docs = []

        # Get the full context titles (Wikipedia article titles)
        context_titles = {ctx[0]: " ".join(ctx[1]) for ctx in question["context"]}
        context_title_to_short = {ctx[0]: ctx[0] for ctx in question["context"]}

        catalog = self._load_catalog()

        for short_title in gold_titles:
            # Find matching full title in context
            full_title = None
            for ctx_title, content in context_titles.items():
                if ctx_title.startswith(short_title) or short_title in ctx_title:
                    full_title = ctx_title
                    break

            if not full_title:
                logger.warning(f"Gold article '{short_title}' not found in question context")
                continue

            if full_title in catalog:
                gold_docs.append({
                    "title": full_title,
                    "text": catalog[full_title]["text"]
                })
            else:
                logger.warning(f"Gold article '{full_title}' not found in catalog")

        return gold_docs

    def _load_distractor_pool(self, qid: str) -> List[str]:
        """Load distractor pool for a specific question (streaming approach)."""
        # For now, load entire file - TODO: implement true streaming
        distractor_streamer = JSONStreamer(str(self.distractor_file))
        distractor_pool = distractor_streamer.get_value_by_key(qid)
        return distractor_pool or []

    def sample_distractors(self, qid: str, gold_titles: List[str], target_tokens: int) -> List[Dict[str, Any]]:
        """Sample distractor articles until token budget is reached."""
        # Load distractor pool for this question
        distractor_pool = self._load_distractor_pool(qid)

        if not distractor_pool:
            logger.warning(f"No distractor pool found for question {qid}")
            return []

        # Use cached catalog
        catalog = self._load_catalog()

        # Shuffle distractors for random sampling
        available_distractors = distractor_pool.copy()
        random.shuffle(available_distractors)

        distractor_docs = []
        current_tokens = 0

        for title in available_distractors:
            if title not in catalog:
                continue

            article = catalog[title]
            article_tokens = article["tokens"]

            # Check if adding this article would exceed budget
            if current_tokens + article_tokens > target_tokens:
                continue

            distractor_docs.append({
                "title": title,
                "text": article["text"]
            })
            current_tokens += article_tokens

            # Stop if we've reached sufficient tokens
            if current_tokens >= target_tokens * 0.9:  # 90% of target
                break

        return distractor_docs

    def create_long_context_record(self, question: Dict[str, Any], context_size: str) -> Dict[str, Any]:
        """Create a long-context record for the specified size."""
        config = self.context_configs[context_size]

        # Extract gold documents
        supporting_docs = self.extract_gold_docs(question)

        if not supporting_docs:
            logger.error(f"No gold documents found for question {question['_id']}")
            return None

        # Sample distractors to reach token target
        gold_titles = [doc["title"] for doc in supporting_docs]
        distractor_docs = self.sample_distractors(
            question["_id"],
            gold_titles,
            config.target_tokens
        )

        # Calculate total tokens
        total_tokens = sum(self.token_counter.count_tokens(doc["text"]) for doc in supporting_docs + distractor_docs)

        # Verify token budget
        if total_tokens < config.min_tokens or total_tokens > config.max_tokens:
            logger.warning(f"Question {question['_id']} context size {total_tokens} tokens (target: {config.target_tokens})")

        return {
            "id": question["_id"],
            "question": question["question"],
            "gold_answer": question["answer"],
            "supporting_docs": supporting_docs,
            "distractor_docs": distractor_docs,
            "context_size": context_size
        }

    def process_all_questions(self, output_dir: str = "data", batch_size: int = 100) -> Dict[str, int]:
        """Process all questions and generate three context size datasets."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Load all questions
        questions = self.load_hotpot_data()

        # Prepare output files
        output_files = {}
        for size in self.context_configs.keys():
            output_files[size] = {
                "path": output_path / f"long_hotpot_{size}.json",
                "records": []
            }

        # Process questions in batches
        total_processed = 0
        total_skipped = 0

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")

            for question in batch:
                processed = False

                for context_size in self.context_configs.keys():
                    record = self.create_long_context_record(question, context_size)
                    if record:
                        output_files[context_size]["records"].append(record)
                        processed = True

                if processed:
                    total_processed += 1
                else:
                    total_skipped += 1

        # Save output files
        for size, data in output_files.items():
            output_file = data["path"]
            records = data["records"]

            logger.info(f"Saving {len(records)} records to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(records, f, indent=2)

            logger.info(f"Saved {len(records)} records ({output_file.stat().st_size} bytes)")

        return {
            "total_processed": total_processed,
            "total_skipped": total_skipped,
            "output_files": list(output_files.keys())
        }

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Process long-context HotpotQA datasets")
    parser.add_argument("--hotpot-file", default="hotpot_dev_fullwiki_v1.json",
                       help="Path to HotpotQA fullwiki file")
    parser.add_argument("--catalog-file", default="data/article_catalog.json",
                       help="Path to article catalog file")
    parser.add_argument("--gold-file", default="data/gold_mappings.json",
                       help="Path to gold mappings file")
    parser.add_argument("--distractor-file", default="data/distractor_pools.json",
                       help="Path to distractor pools file")
    parser.add_argument("--output-dir", default="data",
                       help="Output directory for generated datasets")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing questions")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Process datasets
    processor = LongContextProcessor(
        hotpot_file=args.hotpot_file,
        catalog_file=args.catalog_file,
        gold_file=args.gold_file,
        distractor_file=args.distractor_file
    )

    results = processor.process_all_questions(
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

    logger.info(f"Processing completed: {results}")

if __name__ == "__main__":
    main()
