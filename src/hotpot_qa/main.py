#!/usr/bin/env python3
"""
Main processor for creating long-context HotpotQA datasets.

This script loads HotpotQA full wiki data, fetches supporting articles from Wikipedia,
and exports to JSON format for each context size (8k, 32k, 128k).
"""
import argparse
import logging
import hashlib
from pathlib import Path

from tqdm import tqdm

from .model import ArticleQA, ArticleQAManager, ContextSize
from .utils import RawFullWikiQALoader
from .wikipedia.model import Tokenizer
from .wikipedia.search import WikipediaSearchEngine
from .wikipedia.sqlite_backend import load_dump_to_sqlite

def setup_logging():
    """Set up logging to console with INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="HotpotQA long-context dataset processor")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build SQLite command
    build_parser = subparsers.add_parser("build-sqlite", help="Build SQLite database from Wikipedia dump")
    build_parser.add_argument(
        "--dump-path",
        required=True,
        type=str,
        help="Path to Wikipedia dump BZ2 file (or directory with bz2 files)"
    )
    build_parser.add_argument(
        "--db-path",
        required=True,
        type=str,
        help="Path where SQLite database will be created"
    )
    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for bulk inserts (default: 1000)"
    )
    build_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing database"
    )

    # Process command
    process_parser = subparsers.add_parser("process", help="Process HotpotQA data into long-context datasets")
    process_parser.add_argument(
        "--hotpot-path",
        required=True,
        type=str,
        help="Path to HotpotQA full wiki JSON file (hotpot_dev_fullwiki_v1.json)"
    )
    process_parser.add_argument(
        "--db-path",
        required=True,
        type=str,
        help="Path to SQLite database with Wikipedia articles"
    )
    process_parser.add_argument(
        "--output-dir",
        default="data",
        type=str,
        help="Output directory for JSON files (default: data)"
    )
    process_parser.add_argument(
        "--tokenizer",
        default="simple",
        choices=["simple", "nltk"],
        help="Tokenizer type for text processing (default: simple)"
    )
    process_parser.add_argument(
        "--max-questions",
        type=int,
        help="Limit number of questions to process (for testing)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    setup_logging()
    logger = logging.getLogger(__name__)

    if args.command == "build-sqlite":
        dump_path = Path(args.dump_path)
        db_path = Path(args.db_path)

        if not dump_path.exists():
            logger.error(f"Wikipedia dump not found: {dump_path}")
            return 1

        if db_path.exists() and not args.overwrite:
            logger.error(f"Database already exists: {db_path}. Use --overwrite to replace.")
            return 1

        logger.info("Building SQLite database from Wikipedia dump...")
        try:
            load_dump_to_sqlite(dump_path, db_path, args.batch_size)
            logger.info("Database build complete!")
        except Exception as e:
            logger.error(f"Failed to build database: {e}")
            return 1

        return 0

    elif args.command == "process":
        # Validate inputs
        hotpot_path = Path(args.hotpot_path)
        db_path = Path(args.db_path)
        output_dir = Path(args.output_dir)

        if not hotpot_path.exists():
            logger.error(f"HotpotQA data file not found: {hotpot_path}")
            return 1

        if not db_path.exists():
            logger.error(f"SQLite database not found: {db_path}. Build it with 'build-sqlite' command first.")
            return 1

        logger.info(f"Loading WikipediaSearchEngine from SQLite database: {db_path}")
        tokenizer = Tokenizer(args.tokenizer)
        search_engine = WikipediaSearchEngine(db_path, tokenizer)

        # Build TF-IDF index if not already present
        if not search_engine.index_built:
            logger.info("Building TF-IDF search index (this may take several minutes)...")
            search_engine.build_index()
            logger.info("TF-IDF index built successfully")
        else:
            logger.info("TF-IDF index already available")

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

    else:
        logger.error(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    exit(main())
