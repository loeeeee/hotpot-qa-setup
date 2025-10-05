"""Data analysis utilities for HotpotQA dataset."""

import json
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, List

from hotpot_qa.utils.tokenization import TokenCounter

logger = logging.getLogger(__name__)


class HotpotDataAnalyzer:
    """Comprehensive analyzer for HotpotQA fullwiki dataset."""

    def __init__(self, data_file: str = "hotpot_dev_fullwiki_v1.json"):
        """Initialize analyzer with data file path."""
        self.data_file = Path(data_file)
        self.token_counter = TokenCounter()

    def load_data(self) -> List[Dict]:
        """Load the fullwiki dataset."""
        logger.info(f"Loading data from {self.data_file}")
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} questions")
        return data

    def analyze_structure(self, data: List[Dict]) -> Dict:
        """Analyze the dataset structure."""
        logger.info("Analyzing dataset structure...")

        total_questions = len(data)
        types = Counter()
        levels = Counter()

        # Sample first entry for structure details
        sample = data[0]
        structure_info = {
            "total_questions": total_questions,
            "keys": list(sample.keys()),
            "supporting_facts_length": len(sample["supporting_facts"]),
            "context_length": len(sample["context"])
        }

        for entry in data:
            types[entry.get("type", "unknown")] += 1
            levels[entry.get("level", "unknown")] += 1

        structure_info["types"] = dict(types)
        structure_info["levels"] = dict(levels)

        return structure_info

    def extract_article_catalog(self, data: List[Dict]) -> Dict[str, Dict]:
        """Extract comprehensive article catalog and statistics."""
        logger.info("Extracting article catalog...")

        article_catalog: Dict[str, Dict] = {}
        question_context_sizes: Dict[str, int] = {}

        for entry in data:
            qid = entry["_id"]

            # Process context articles
            context_tokens = 0
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

                article_catalog[title]["questions_appearing_in"].append(qid)
                context_tokens += article_tokens

            question_context_sizes[qid] = context_tokens

        logger.info(f"Extracted {len(article_catalog)} unique articles")
        return {
            "article_catalog": article_catalog,
            "question_context_sizes": question_context_sizes
        }

    def analyze_context_sizes(self, question_context_sizes: Dict[str, int]) -> Dict:
        """Analyze context size distributions."""
        sizes = list(question_context_sizes.values())
        sorted_sizes = sorted(sizes)

        return {
            "min_tokens": min(sizes),
            "max_tokens": max(sizes),
            "mean_tokens": sum(sizes) / len(sizes),
            "median_tokens": sorted_sizes[len(sizes) // 2],
            "p25_tokens": sorted_sizes[len(sizes) // 4],
            "p75_tokens": sorted_sizes[3 * len(sizes) // 4],
            "p95_tokens": sorted_sizes[int(0.95 * len(sizes))]
        }

    def save_analysis(self, analysis_results: Dict, output_file: str) -> None:
        """Save analysis results to JSON file."""
        output_path = Path(output_file)
        logger.info(f"Saving analysis to {output_path}")

        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)

        logger.info(f"Analysis saved ({output_path.stat().st_size} bytes)")


    def run_analysis(self, output_file: str = "data/analysis_results.json") -> Dict:
        """Run complete analysis pipeline."""
        logger.info("Starting HotpotQA data analysis...")

        # Load data
        data = self.load_data()

        # Structure analysis
        structure = self.analyze_structure(data)

        # Article catalog extraction
        catalog_data = self.extract_article_catalog(data)

        # Context size analysis
        context_sizes = self.analyze_context_sizes(catalog_data["question_context_sizes"])

        # Combine results
        analysis_results = {
            "structure": structure,
            "context_sizes": context_sizes,
            "catalog_stats": {
                "total_unique_articles": len(catalog_data["article_catalog"]),
                "total_questions": len(catalog_data["question_context_sizes"])
            }
        }

        # Save results
        self.save_analysis(analysis_results, output_file)

        logger.info("Data analysis completed successfully")
        return analysis_results


def main():
    """Main entry point for CLI."""
    analyzer = HotpotDataAnalyzer()
    results = analyzer.run_analysis()


if __name__ == "__main__":
    main()
