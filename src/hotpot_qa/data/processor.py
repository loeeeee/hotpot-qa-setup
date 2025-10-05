"""
Long Context HotpotQA Dataset Processor
Creates datasets with extended contexts by adding distractor articles.
"""

import json
import logging
import random
import sqlite3
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from hotpot_qa.utils.json_stream import JSONStreamer
from hotpot_qa.utils.tokenization import TokenCounter

logger = logging.getLogger(__name__)


class DistractorPoolStore:
    """
    Provides fast random access to distractor pools backed by an on-disk SQLite cache.

    The original JSON file is ~13 GB and structured as a single giant dictionary. Reading it
    repeatedly for each HotpotQA question results in catastrophic performance. This store builds
    (or reuses) a compressed SQLite database that enables O(log n) lookups with negligible memory
    overhead.
    """

    def __init__(self, json_path: Path):
        self.json_path = Path(json_path)
        self.db_path = self.json_path.with_suffix(".sqlite3")
        self._conn: Optional[sqlite3.Connection] = None
        self._prepare_store()

    def _prepare_store(self) -> None:
        """Ensure the SQLite cache exists and is up to date."""
        rebuild = True
        if self.db_path.exists():
            rebuild = self.db_path.stat().st_mtime < self.json_path.stat().st_mtime

        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode = OFF")
        self._conn.execute("PRAGMA synchronous = OFF")
        self._conn.execute("PRAGMA temp_store = MEMORY")

        if not rebuild and self._table_exists():
            logger.info("Using cached distractor pools at %s", self.db_path)
            return

        logger.info("Building distractor pool cache at %s", self.db_path)
        self._conn.execute("DROP TABLE IF EXISTS pools")
        self._conn.execute("CREATE TABLE pools (qid TEXT PRIMARY KEY, payload BLOB)")

        try:
            import ijson  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
            raise RuntimeError(
                "ijson is required to build the distractor pool cache"
            ) from exc

        cursor = self._conn.cursor()
        batch: List[Tuple[str, bytes]] = []

        with self._conn:
            with open(self.json_path, "rb") as fh:
                progress = tqdm(desc="Indexing distractor pools", unit="pool", leave=False)
                try:
                    for qid, titles in ijson.kvitems(fh, ""):
                        payload = zlib.compress(json.dumps(titles).encode("utf-8"))
                        batch.append((qid, payload))

                        if len(batch) >= 256:
                            cursor.executemany("INSERT INTO pools (qid, payload) VALUES (?, ?)", batch)
                            batch.clear()

                        progress.update(1)

                    if batch:
                        cursor.executemany("INSERT INTO pools (qid, payload) VALUES (?, ?)", batch)
                finally:
                    progress.close()

        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_pools_qid ON pools(qid)")
        logger.info("Finished building distractor pool cache")

    def _table_exists(self) -> bool:
        assert self._conn is not None
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pools'"
        )
        return cursor.fetchone() is not None

    @lru_cache(maxsize=2048)
    def _cached_get(self, qid: str) -> List[str]:
        assert self._conn is not None
        cursor = self._conn.execute("SELECT payload FROM pools WHERE qid = ?", (qid,))
        row = cursor.fetchone()
        if row is None:
            logger.warning("No distractor pool found for question %s", qid)
        return [] if row is None else json.loads(zlib.decompress(row[0]).decode("utf-8"))

    def get(self, qid: str) -> List[str]:
        return self._cached_get(qid)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._cached_get.cache_clear()


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

    def __init__(
        self,
        hotpot_file: str = "hotpot_dev_fullwiki_v1.json",
        catalog_file: str = "data/article_catalog.json",
        gold_file: str = "data/gold_mappings.json",
        distractor_file: str = "data/distractor_pools.json",
    ):
        self.hotpot_file = Path(hotpot_file)
        self.catalog_file = Path(catalog_file)
        self.gold_file = Path(gold_file)
        self.distractor_file = Path(distractor_file)

        # Initialize utilities
        self.token_counter = TokenCounter()
        self._rng = random.Random(42)

        # Context configurations
        self.context_configs = {
            "8k": ContextConfig("8k", 8000),
            "32k": ContextConfig("32k", 32000),
            "128k": ContextConfig("128k", 128000),
        }

        # Cache for loaded resources
        self._catalog_cache: Optional[Dict[str, Any]] = None
        self._gold_cache: Optional[Dict[str, List[str]]] = None
        self._distractor_store = DistractorPoolStore(self.distractor_file)

    def load_hotpot_data(self) -> List[Dict[str, Any]]:
        """Load HotpotQA dataset."""
        logger.info("Loading HotpotQA data from %s", self.hotpot_file)
        with open(self.hotpot_file, "r") as f:
            data = json.load(f)
        logger.info("Loaded %d questions", len(data))
        return data

    def _load_catalog(self) -> Dict[str, Any]:
        """Load article catalog (cached)."""
        if self._catalog_cache is None:
            catalog_streamer = JSONStreamer(str(self.catalog_file))
            self._catalog_cache = catalog_streamer.load_small_file()
            logger.info("Loaded catalog with %d articles", len(self._catalog_cache))
        return self._catalog_cache

    def _load_gold_mappings(self) -> Dict[str, List[str]]:
        if self._gold_cache is None:
            with open(self.gold_file, "r") as fh:
                mappings = json.load(fh)
            # Normalize to preserve order while removing duplicates
            normalized: Dict[str, List[str]] = {}
            for qid, titles in mappings.items():
                seen = set()
                ordered = []
                for title in titles:
                    if title not in seen:
                        ordered.append(title)
                        seen.add(title)
                normalized[qid] = ordered
            self._gold_cache = normalized
            logger.info("Loaded gold mappings for %d questions", len(self._gold_cache))
        return self._gold_cache

    def _get_article_entry(self, title: str) -> Optional[Dict[str, Any]]:
        catalog = self._load_catalog()
        entry = catalog.get(title)
        if entry is None:
            logger.warning("Article '%s' not found in catalog", title)
        return entry

    def extract_gold_docs(self, question: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        """Extract gold documents with full text and aggregated token counts."""
        gold_titles = self._load_gold_mappings().get(question["_id"], [])
        gold_docs: List[Dict[str, Any]] = []
        total_tokens = 0

        for title in gold_titles:
            article = self._get_article_entry(title)
            if not article:
                continue
            gold_docs.append({"title": title, "text": article["text"]})
            total_tokens += int(article.get("tokens") or self.token_counter.count_tokens(article["text"]))

        if not gold_docs:
            logger.error("No gold documents found for question %s", question["_id"])

        return gold_docs, total_tokens

    def sample_distractors(
        self, qid: str, gold_titles: List[str], target_tokens: int
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Sample distractor articles until token budget is reached."""
        distractor_candidates = self._distractor_store.get(qid)
        if not distractor_candidates:
            logger.warning("No distractor pool found for question %s", qid)
            return [], 0

        catalog = self._load_catalog()
        local_rng = self._rng
        shuffled = distractor_candidates.copy()
        local_rng.shuffle(shuffled)

        distractor_docs: List[Dict[str, Any]] = []
        current_tokens = 0
        max_tokens = target_tokens * 1.05  # small elastic buffer

        for title in shuffled:
            if title in gold_titles:
                continue

            article = catalog.get(title)
            if article is None:
                continue

            article_tokens = int(article.get("tokens") or self.token_counter.count_tokens(article["text"]))
            if current_tokens + article_tokens > max_tokens:
                continue

            distractor_docs.append({"title": title, "text": article["text"]})
            current_tokens += article_tokens

            if current_tokens >= target_tokens * 0.95:
                break

        return distractor_docs, current_tokens

    def create_long_context_record(self, question: Dict[str, Any], context_size: str) -> Optional[Dict[str, Any]]:
        """Create a long-context record for the specified size."""
        config = self.context_configs[context_size]

        supporting_docs, supporting_tokens = self.extract_gold_docs(question)
        if not supporting_docs:
            return None

        gold_titles = [doc["title"] for doc in supporting_docs]
        distractor_docs, distractor_tokens = self.sample_distractors(
            question["_id"], gold_titles, config.target_tokens
        )

        total_tokens = supporting_tokens + distractor_tokens

        if total_tokens < config.min_tokens or total_tokens > config.max_tokens:
            logger.warning(
                "Question %s context size %d tokens (target: %d)",
                question["_id"],
                total_tokens,
                config.target_tokens,
            )

        return {
            "id": question["_id"],
            "question": question["question"],
            "gold_answer": question["answer"],
            "supporting_docs": supporting_docs,
            "distractor_docs": distractor_docs,
            "context_size": context_size,
        }

    def process_all_questions(self, output_dir: str = "data", batch_size: int = 100) -> Dict[str, Any]:
        """Process all questions and generate three context size datasets."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        questions = self.load_hotpot_data()
        total_questions = len(questions)

        output_files: Dict[str, Dict[str, Any]] = {}
        for size in self.context_configs.keys():
            output_files[size] = {
                "path": output_path / f"long_hotpot_{size}.json",
                "records": [],
            }

        total_processed = 0
        total_skipped = 0
        total_batches = (total_questions + batch_size - 1) // batch_size if batch_size > 0 else 0

        if total_questions == 0:
            logger.warning("No questions found to process.")
        else:
            progress_bar = tqdm(total=total_questions, desc="Processing questions", unit="question", leave=False)
            try:
                for batch_idx in range(0, total_questions, batch_size):
                    batch = questions[batch_idx : batch_idx + batch_size]
                    logger.info("Processing batch %d/%d", (batch_idx // batch_size) + 1, total_batches)

                    for question in batch:
                        processed = False

                        for context_size in self.context_configs.keys():
                            record = self.create_long_context_record(question, context_size)
                            if record is not None:
                                output_files[context_size]["records"].append(record)
                                processed = True

                        if processed:
                            total_processed += 1
                        else:
                            total_skipped += 1

                        progress_bar.update(1)
            finally:
                if total_questions > 0:
                    progress_bar.close()

        for size, data in output_files.items():
            output_file = data["path"]
            records = data["records"]

            logger.info("Saving %d records to %s", len(records), output_file)
            with open(output_file, "w") as f:
                json.dump(records, f, indent=2)

            logger.info("Saved %d records (%.2f MB)", len(records), output_file.stat().st_size / (1024 * 1024))

        # Clean up heavy resources
        self._distractor_store.close()

        return {
            "total_processed": total_processed,
            "total_skipped": total_skipped,
            "output_files": list(output_files.keys()),
        }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Process long-context HotpotQA datasets")
    parser.add_argument("--hotpot-file", default="hotpot_dev_fullwiki_v1.json", help="Path to HotpotQA fullwiki file")
    parser.add_argument("--catalog-file", default="data/article_catalog.json", help="Path to article catalog file")
    parser.add_argument("--gold-file", default="data/gold_mappings.json", help="Path to gold mappings file")
    parser.add_argument("--distractor-file", default="data/distractor_pools.json", help="Path to distractor pools file")
    parser.add_argument("--output-dir", default="data", help="Output directory for generated datasets")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing questions")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    processor = LongContextProcessor(
        hotpot_file=args.hotpot_file,
        catalog_file=args.catalog_file,
        gold_file=args.gold_file,
        distractor_file=args.distractor_file,
    )

    results = processor.process_all_questions(output_dir=args.output_dir, batch_size=args.batch_size)
    logger.info("Processing completed: %s", results)


if __name__ == "__main__":
    main()
