#!/usr/bin/env python3
"""
Main processor for creating long-context HotpotQA datasets.

This script loads HotpotQA full wiki data, fetches supporting articles from Wikipedia,
and exports to JSON format for each context size (8k, 32k, 128k).
"""
import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from .model import ArticleQA, ArticleQAManager, ContextSize
from .utils import RawFullWikiQALoader
from .wikipedia.model import Wikipedia, Tokenizer
from .wikipedia.search import WikipediaSearchEngine

def setup_logging():
    """Set up logging to console with INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="Process HotpotQA full wiki data into long-context datasets")
    parser.add_argument(
        "--hotpot-path",
        required=True,
        type=str,
        help="Path to HotpotQA full wiki JSON file (hotpot_dev_fullwiki_v1.json)"
    )
    parser.add_argument(
        "--wikipedia-path",
        required=True,
        type=str,
        help="Path to Wikipedia dump BZ2 file (enwiki-..-processed.tar.bz2)"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        type=str,
        help="Output directory for JSON files (default: data)"
    )
    parser.add_argument(
        "--tokenizer",
        default="simple",
        choices=["simple", "nltk"],
        help="Tokenizer type for text processing (default: simple)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Limit number of questions to process (for testing)"
    )

    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # Validate inputs
    hotpot_path = Path(args.hotpot_path)
    wikipedia_path = Path(args.wikipedia_path)
    output_dir = Path(args.output_dir)

    if not hotpot_path.exists():
        logger.error(f"HotpotQA data file not found: {hotpot_path}")
        return 1

    if not wikipedia_path.exists():
        logger.error(f"Wikipedia dump file not found: {wikipedia_path}")
        return 1

    logger.info("Loading Wikipedia dump... (this may take several minutes)")
    try:
        wikipedia = Wikipedia.from_bz(wikipedia_path)
        logger.info(f"Loaded {len(wikipedia.articles)} Wikipedia articles")
    except Exception as e:
        logger.error(f"Failed to load Wikipedia dump: {e}")
        return 1

    tokenizer = Tokenizer(args.tokenizer)
    search_engine = WikipediaSearchEngine(wikipedia, tokenizer)

    logger.info("Loading HotpotQA data...")
    loader = RawFullWikiQALoader(hotpot_path)
    raw_qas = list(loader)
    logger.info(f"Loaded {len(raw_qas)} HotpotQA questions")

    if args.max_questions:
        raw_qas = raw_qas[:args.max_questions]
        logger.info(f"Limited to {len(raw_qas)} questions for testing")

    # Convert to ArticleQA
    manager = ArticleQAManager()
    successful_conversions = 0

    logger.info("Converting to ArticleQA with supporting docs...")
    for raw_qa in tqdm(raw_qas, desc="Processing QAs"):
        try:
            qa = ArticleQA.from_raw_full_wiki(raw_qa, search_engine)
            manager.add(qa)
            successful_conversions += 1
        except Exception as e:
            logger.warning(f"Failed to process QA {raw_qa.get('_id', 'unknown')}: {e}")

    logger.info(f"Successfully processed {successful_conversions}/{len(raw_qas)} questions")

    # Export to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting to {output_dir}")
    manager.to_json_per_size(output_dir)

    logger.info("Processing complete!")
    logger.info(f"Output files: {[f'long_hotpot_{size.value}k.json' for size in ContextSize]}")

    return 0

if __name__ == "__main__":
    exit(main())
