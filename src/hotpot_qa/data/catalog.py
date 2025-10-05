"""Article catalog extraction utilities for HotpotQA dataset."""

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

from hotpot_qa.utils.tokenization import TokenCounter

logger = logging.getLogger(__name__)


class ArticleCatalogExtractor:
    """Extracts article catalog from HotpotQA fullwiki for distractor sampling."""

    def __init__(self, data_file: str = "hotpot_dev_fullwiki_v1.json"):
        """Initialize extractor with data file path."""
        self.data_file = Path(data_file)
        self.token_counter = TokenCounter()

    def load_data(self) -> List[Dict]:
        """Load the fullwiki dataset."""
        logger.info(f"Loading data from {self.data_file}")
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        return data

    def build_article_catalog(self, data: List[Dict]) -> Dict[str, Dict]:
        """Build complete catalog of all articles in fullwiki contexts."""
        logger.info("Building comprehensive article catalog...")

        article_catalog: Dict[str, Dict] = {}
        question_count = 0

        for entry in data:
            question_count += 1
            if question_count % 1000 == 0:
                logger.info(f"Processed {question_count} questions...")

            for title, sentences in entry["context"]:
                # Concatenate sentences into full text
                full_text = " ".join(sentences)
                article_tokens = self.token_counter.count_tokens(full_text)

                if title not in article_catalog:
                    article_catalog[title] = {
                        "text": full_text,
                        "tokens": article_tokens,
                        "questions_appearing_in": []
                    }

                article_catalog[title]["questions_appearing_in"].append(entry["_id"])

        # Sort by question count (most frequent first)
        sorted_catalog = {k: v for k, v in sorted(
            article_catalog.items(),
            key=lambda x: len(x[1]["questions_appearing_in"]),
            reverse=True
        )}

        logger.info(f"Built catalog with {len(sorted_catalog)} unique articles")
        return sorted_catalog

    def extract_gold_mappings(self, data: List[Dict]) -> Dict[str, List[str]]:
        """Extract gold article mappings per question."""
        logger.info("Extracting gold article mappings...")

        gold_mappings: Dict[str, List[str]] = {}

        for entry in data:
            qid = entry["_id"]
            gold_articles = []
            for fact in entry["supporting_facts"]:
                gold_articles.append(fact[0])

            gold_mappings[qid] = gold_articles

        logger.info(f"Extracted gold mappings for {len(gold_mappings)} questions")
        return gold_mappings

    def create_distractor_pools(self, article_catalog: Dict[str, Dict],
                               gold_mappings: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Create distractor article pools for each question (excluding gold articles)."""
        logger.info("Creating distractor pools for all questions...")

        distractor_pools: Dict[str, List[str]] = {}
        all_articles = set(article_catalog.keys())

        for qid, gold_articles in gold_mappings.items():
            gold_set = set(gold_articles)
            available_distractors = list(all_articles - gold_set)
            distractor_pools[qid] = available_distractors

        logger.info(f"Created distractor pools for {len(distractor_pools)} questions")
        return distractor_pools

    def save_catalog(self, catalog: Dict[str, Dict], output_file: str) -> None:
        """Save article catalog to JSON file."""
        output_path = Path(output_file)
        logger.info(f"Saving article catalog to {output_path}")

        with open(output_path, 'w') as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"Article catalog saved ({output_path.stat().st_size} bytes)")

    def save_mappings(self, mappings: Dict, output_file: str) -> None:
        """Save mappings to JSON file."""
        output_path = Path(output_file)
        logger.info(f"Saving mappings to {output_path}")

        with open(output_path, 'w') as f:
            json.dump(mappings, f, indent=2)

        logger.info(f"Mappings saved ({output_path.stat().st_size} bytes)")

    def run_extraction(self,
                       catalog_file: str = "data/article_catalog.json",
                       gold_file: str = "data/gold_mappings.json",
                       distractor_file: str = "data/distractor_pools.json") -> Dict:
        """Run complete extraction pipeline."""
        logger.info("Starting article catalog extraction...")

        # Load data
        data = self.load_data()

        # Build article catalog
        article_catalog = self.build_article_catalog(data)

        # Extract gold mappings
        gold_mappings = self.extract_gold_mappings(data)

        # Create distractor pools
        distractor_pools = self.create_distractor_pools(article_catalog, gold_mappings)

        # Save outputs
        self.save_catalog(article_catalog, catalog_file)
        self.save_mappings(gold_mappings, gold_file)
        self.save_mappings(distractor_pools, distractor_file)

        results = {
            "catalog_stats": {
                "total_articles": len(article_catalog),
                "total_questions": len(gold_mappings)
            },
            "output_files": [catalog_file, gold_file, distractor_file]
        }

        logger.info("Article catalog extraction completed successfully")
        return results


def main():
    """Main entry point for CLI."""
    extractor = ArticleCatalogExtractor()
    results = extractor.run_extraction()


if __name__ == "__main__":
    main()
