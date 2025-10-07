import logging
import sqlite3
import json
import bz2
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Tuple, TYPE_CHECKING, Sequence

import multiprocessing
import tarfile
from threading import local as thread_local
from queue import Empty, Full

from tqdm import tqdm

if TYPE_CHECKING:
    from .model import WikipediaArticle

logger = logging.getLogger(__name__)

SerializedArticleRow = Tuple[str, str, Optional[str], str, str, str, int]

INSERT_ARTICLE_SQL = """
    INSERT OR REPLACE INTO articles
    (id, title, url, paragraphs_json, token_json, links_json, total_token)
    VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_WORKER_BATCH = "batch"
_WORKER_FILE_DONE = "file_done"
_WORKER_ERROR = "error"
_WORKER_DONE = "done"


def _safe_queue_put(queue, item, stop_event) -> bool:
    """
    Attempt to put an item onto a multiprocessing queue while respecting a stop event.

    Returns False if the stop_event becomes set before the item can be queued.
    """
    while True:
        if stop_event.is_set():
            return False
        try:
            queue.put(item, timeout=1.0)
            return True
        except Full:
            if stop_event.is_set():
                return False
            continue


def _serialize_article_for_db(article: "WikipediaArticle") -> SerializedArticleRow:
    """Convert a WikipediaArticle into a row tuple suitable for SQLite executemany."""
    return (
        article.id,
        article.title,
        None,  # URL is not currently available
        json.dumps(article.paragraphs),
        json.dumps(article.token),
        json.dumps(article.links),
        article.total_token,
    )


def _worker_parse_bz2(task_queue, result_queue, chunk_size: int, stop_event) -> None:
    """Worker process that parses bz2 files and streams serialized article rows."""
    from .model import WikipediaArticle, TokenCounter

    WikipediaArticle.token_counter = TokenCounter(method="gpt")

    notified_done = False

    while not stop_event.is_set():
        try:
            path_str = task_queue.get(timeout=1.0)
        except Empty:
            continue

        if path_str is None:
            if _safe_queue_put(result_queue, (_WORKER_DONE, None, None), stop_event):
                notified_done = True
            break

        path = Path(path_str)
        article_count = 0
        chunk: List[SerializedArticleRow] = []

        try:
            with bz2.open(path, "rt", encoding="utf-8", errors="replace") as source:
                for line in source:
                    if stop_event.is_set():
                        break
                    if not line.strip():
                        continue
                    try:
                        article = WikipediaArticle.from_json(line)
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON lines

                    chunk.append(_serialize_article_for_db(article))
                    article_count += 1

                    if len(chunk) >= chunk_size:
                        if not _safe_queue_put(result_queue, (_WORKER_BATCH, chunk), stop_event):
                            chunk = []
                            break
                        chunk = []

            if stop_event.is_set():
                break

            if chunk:
                if not _safe_queue_put(result_queue, (_WORKER_BATCH, chunk), stop_event):
                    break

            if not _safe_queue_put(result_queue, (_WORKER_FILE_DONE, path_str, article_count), stop_event):
                break
        except Exception as exc:  # noqa: BLE001
            if not _safe_queue_put(result_queue, (_WORKER_ERROR, path_str, str(exc)), stop_event):
                break

    if not notified_done:
        _safe_queue_put(result_queue, (_WORKER_DONE, None, None), stop_event)


@dataclass
class WikipediaSQLiteConfig:
    db_path: Path
    batch_size: int = 1000
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"


@dataclass
class WikipediaSQLiteIndex:
    config: WikipediaSQLiteConfig
    _connection_local: thread_local = field(init=False)

    def __post_init__(self):
        self._connection_local = thread_local()
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a process-local SQLite connection with proper configuration."""
        if not hasattr(self._connection_local, "connection"):
            conn = sqlite3.connect(self.config.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            conn.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
            conn.execute(f"PRAGMA synchronous = {self.config.synchronous}")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory map
            conn.execute("PRAGMA cache_size = 1000")  # 1MB cache

            self._connection_local.connection = conn

        return self._connection_local.connection

    def _ensure_schema(self) -> None:
        """Create the database schema if it doesn't exist."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT UNIQUE COLLATE NOCASE,
                    url TEXT,
                    paragraphs_json TEXT NOT NULL,
                    token_json TEXT NOT NULL,
                    links_json TEXT NOT NULL,
                    total_token INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_title ON articles(title COLLATE NOCASE)
                """
            )

    def _serialize_article(self, article: "WikipediaArticle") -> SerializedArticleRow:
        """Serialize a WikipediaArticle to a tuple of database fields."""
        return _serialize_article_for_db(article)

    def _deserialize_row(self, row: sqlite3.Row) -> "WikipediaArticle":
        """Deserialize a database row to WikipediaArticle."""
        from . import model  # Import here to avoid circular import

        return model.WikipediaArticle(
            id=row["id"],
            title=row["title"],
            paragraphs=json.loads(row["paragraphs_json"]),
            token=json.loads(row["token_json"]),
            links=json.loads(row["links_json"]),
        )

    def begin_transaction(self) -> None:
        """Begin an explicit transaction."""
        conn = self._get_connection()
        conn.execute("BEGIN IMMEDIATE")

    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        conn = self._get_connection()
        conn.commit()

    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        conn = self._get_connection()
        conn.rollback()

    def executemany_insert(self, rows: Sequence[SerializedArticleRow]) -> None:
        """Execute an INSERT executemany using prepared rows without committing."""
        if not rows:
            return
        conn = self._get_connection()
        conn.executemany(INSERT_ARTICLE_SQL, rows)

    def bulk_insert(self, articles: Sequence["WikipediaArticle"]) -> None:
        """Insert multiple articles in a single transaction."""
        if not articles:
            return

        rows = [self._serialize_article(article) for article in articles]

        self.begin_transaction()
        try:
            self.executemany_insert(rows)
        except Exception:
            self.rollback_transaction()
            raise
        else:
            self.commit_transaction()

        logger.info("Inserted %d articles into SQLite database", len(articles))

    def get_article_by_title(self, title: str) -> Optional["WikipediaArticle"]:
        """Retrieve a single article by exact title match."""
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM articles WHERE title = ? COLLATE NOCASE
            """,
            (title,),
        )
        row = cursor.fetchone()
        return self._deserialize_row(row) if row else None

    def get_random_articles(self, limit: int, exclude_titles: Set[str]) -> List["WikipediaArticle"]:
        """Get random articles excluding specified titles."""
        if not limit:
            return []

        exclude_placeholders = ",".join("?" for _ in exclude_titles)
        exclude_condition = f"WHERE title NOT IN ({exclude_placeholders})" if exclude_titles else ""

        conn = self._get_connection()
        cursor = conn.execute(
            f"""
            SELECT * FROM articles {exclude_condition}
            ORDER BY RANDOM() LIMIT ?
            """,
            list(exclude_titles) + [limit],
        )

        return [self._deserialize_row(row) for row in cursor.fetchall()]

    def get_article_count(self) -> int:
        """Get total number of articles in the database."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM articles")
        return cursor.fetchone()[0]

    def get_all_titles(self) -> List[str]:
        """Get all article titles (for debugging/validation)."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT title FROM articles ORDER BY title")
        return [row["title"] for row in cursor.fetchall()]

    def exists(self, title: str) -> bool:
        """Check if an article exists."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM articles WHERE title = ? COLLATE NOCASE LIMIT 1", (title,)
        )
        return cursor.fetchone() is not None


def load_dump_to_sqlite(dump_path: Path, db_path: Path, batch_size: int = 1000) -> None:
    """
    Load Wikipedia dump into SQLite database.

    Args:
        dump_path: Path to Wikipedia dump (tar.bz2 file or directory with bz2 files)
        db_path: Path where SQLite database will be created
        batch_size: Batch size for bulk inserts
    """
    config = WikipediaSQLiteConfig(db_path=db_path, batch_size=batch_size)
    index = WikipediaSQLiteIndex(config)

    base_path = dump_path
    temp_extract_dir = None

    # Handle tar.bz2 file - extract to temporary directory
    if dump_path.suffix == ".bz2" and tarfile.is_tarfile(dump_path):
        logger.info("Detected tar.bz2 file, extracting to temporary directory...")
        temp_extract_dir = Path(tempfile.mkdtemp())
        with tarfile.open(dump_path, "r:bz2") as tar:
            tar.extractall(temp_extract_dir)
        base_path = temp_extract_dir
    elif dump_path.is_dir():
        logger.info("Detected extracted directory, using directly...")
    else:
        raise ValueError(f"Path {dump_path} is neither a valid tar.bz2 file nor a directory")

    # Find all .bz2 files in the base directory
    bz2_files = [f for f in base_path.rglob("*.bz2") if f.is_file()]

    logger.info("Found %d bz2 files to process", len(bz2_files))

    if not bz2_files:
        raise ValueError(f"No bz2 files found in {base_path}")

    max_processes = max(1, min(multiprocessing.cpu_count() or 1, len(bz2_files)))
    logger.info("Using %d parallel processes for loading", max_processes)

    worker_chunk_size = max(64, batch_size // 2) or 64

    ctx = multiprocessing.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max_processes * 2)
    result_queue = ctx.Queue(maxsize=max_processes * 4)
    stop_event = ctx.Event()

    workers = []
    for _ in range(max_processes):
        proc = ctx.Process(
            target=_worker_parse_bz2,
            args=(task_queue, result_queue, worker_chunk_size, stop_event),
        )
        proc.daemon = True
        proc.start()
        workers.append(proc)

    completed_workers = 0
    error_count = 0
    total_articles = 0
    articles_in_transaction = 0
    transaction_active = False
    commit_threshold = max(batch_size * 8, 5000)
    buffer: List[SerializedArticleRow] = []
    interrupted = False

    article_bar = tqdm(total=None, desc="Inserted articles", unit="article")
    file_bar = tqdm(total=len(bz2_files), desc="Processed bz2 files", unit="file")

    def ensure_transaction() -> None:
        nonlocal transaction_active
        if not transaction_active:
            index.begin_transaction()
            transaction_active = True

    def flush_buffer(force: bool = False) -> None:
        nonlocal buffer, articles_in_transaction, transaction_active, total_articles
        while buffer and (force or len(buffer) >= batch_size):
            ensure_transaction()
            take = batch_size if not force else len(buffer)
            rows = buffer[:take]
            index.executemany_insert(rows)
            total_articles += len(rows)
            if article_bar is not None:
                article_bar.update(len(rows))
            articles_in_transaction += len(rows)
            del buffer[:take]

            if articles_in_transaction >= commit_threshold:
                index.commit_transaction()
                transaction_active = False
                articles_in_transaction = 0

        if force and transaction_active:
            index.commit_transaction()
            transaction_active = False
            articles_in_transaction = 0

    try:
        try:
            for file_path in bz2_files:
                while True:
                    try:
                        task_queue.put(str(file_path), timeout=1.0)
                        break
                    except Full:
                        if stop_event.is_set():
                            raise KeyboardInterrupt
            for _ in workers:
                task_queue.put(None)
        except KeyboardInterrupt:
            interrupted = True
            stop_event.set()
            logger.warning("Import interrupted while scheduling tasks...")

        if not interrupted:
            try:
                while completed_workers < len(workers):
                    try:
                        message_type, payload, extra = result_queue.get(timeout=1.0)
                    except Empty:
                        if stop_event.is_set() and completed_workers >= len(workers):
                            break
                        continue

                    if message_type == _WORKER_BATCH:
                        rows = payload or []
                        buffer.extend(rows)
                        flush_buffer()
                    elif message_type == _WORKER_FILE_DONE:
                        if file_bar is not None:
                            file_bar.update(1)
                    elif message_type == _WORKER_ERROR:
                        logger.warning("Error processing %s: %s", payload, extra)
                        error_count += 1
                    elif message_type == _WORKER_DONE:
                        completed_workers += 1
                    else:
                        logger.error("Unknown message type received from worker: %s", message_type)
            except KeyboardInterrupt:
                interrupted = True
                stop_event.set()
                logger.warning("Import interrupted by user. Finalizing current work...")
    finally:
        stop_event.set()
        flush_buffer(force=True)
        if article_bar is not None:
            article_bar.close()
        if file_bar is not None:
            file_bar.close()
        if transaction_active:
            index.commit_transaction()

        for proc in workers:
            proc.join()

    task_queue.close()
    task_queue.join_thread()
    result_queue.close()
    result_queue.join_thread()

    if temp_extract_dir:
        shutil.rmtree(temp_extract_dir)

    logger.info(
        "Ingestion complete! Total articles inserted: %d, errors: %d", total_articles, error_count
    )

    if error_count > 0:
        logger.warning("There were %d errors during processing", error_count)

    if interrupted:
        raise KeyboardInterrupt
