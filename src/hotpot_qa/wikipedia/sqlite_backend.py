import logging
import sqlite3
import json
import bz2
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Tuple, TYPE_CHECKING
import multiprocessing.util
import multiprocessing
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import local as thread_local
from tqdm import tqdm

if TYPE_CHECKING:
    from .model import WikipediaArticle

logger = logging.getLogger(__name__)


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
        if not hasattr(self._connection_local, 'connection'):
            conn = sqlite3.connect(self.config.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            conn.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
            conn.execute(f"PRAGMA synchronous = {self.config.synchronous}")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB memory map
            conn.execute("PRAGMA cache_size = 1000")  # 1MB cache

            # Disable connection check in multiprocessing context
            # Connections must be created and used in the same process
            self._connection_local.connection = conn

        return self._connection_local.connection

    def _ensure_schema(self) -> None:
        """Create the database schema if it doesn't exist."""
        conn = self._get_connection()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT UNIQUE COLLATE NOCASE,
                    url TEXT,
                    paragraphs_json TEXT NOT NULL,
                    token_json TEXT NOT NULL,
                    links_json TEXT NOT NULL,
                    total_token INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_title ON articles(title COLLATE NOCASE)
            """)

    def _serialize_article(self, article: "WikipediaArticle") -> Dict[str, Any]:
        """Serialize a WikipediaArticle to database fields."""
        return {
            'id': article.id,
            'title': article.title,
            'url': None,  # Not available in current model
            'paragraphs_json': json.dumps(article.paragraphs),
            'token_json': json.dumps(article.token),
            'links_json': json.dumps(article.links),
            'total_token': article.total_token
        }

    def _deserialize_row(self, row: sqlite3.Row) -> "WikipediaArticle":
        """Deserialize a database row to WikipediaArticle."""
        # Import here to avoid circular import
        from . import model
        return model.WikipediaArticle(
            id=row['id'],
            title=row['title'],
            paragraphs=json.loads(row['paragraphs_json']),
            token=json.loads(row['token_json']),
            links=json.loads(row['links_json'])
        )

    def bulk_insert(self, articles: List["WikipediaArticle"]) -> None:
        """Insert multiple articles in batches."""
        conn = self._get_connection()

        chunk_size = self.config.batch_size
        total_inserted = 0

        for i in range(0, len(articles), chunk_size):
            chunk = articles[i:i + chunk_size]
            serialized = [self._serialize_article(article) for article in chunk]

            with conn:
                try:
                    conn.executemany("""
                        INSERT OR REPLACE INTO articles
                        (id, title, url, paragraphs_json, token_json, links_json, total_token)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, [
                        (art['id'], art['title'], art['url'], art['paragraphs_json'], art['token_json'], art['links_json'], art['total_token'])
                        for art in serialized
                    ])
                    total_inserted += len(chunk)
                except Exception as e:
                    logger.error(f"Error inserting batch {i // chunk_size + 1}: {e}")
                    raise

        logger.info(f"Inserted {total_inserted} articles into SQLite database")

    def get_article_by_title(self, title: str) -> Optional["WikipediaArticle"]:
        """Retrieve a single article by exact title match."""
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM articles WHERE title = ? COLLATE NOCASE
        """, (title,))
        row = cursor.fetchone()
        return self._deserialize_row(row) if row else None

    def get_random_articles(self, limit: int, exclude_titles: Set[str]) -> List["WikipediaArticle"]:
        """Get random articles excluding specified titles."""
        if not limit:
            return []

        exclude_placeholders = ','.join('?' for _ in exclude_titles)
        exclude_condition = f"WHERE title NOT IN ({exclude_placeholders})" if exclude_titles else ""

        conn = self._get_connection()
        cursor = conn.execute(f"""
            SELECT * FROM articles {exclude_condition}
            ORDER BY RANDOM() LIMIT ?
        """, list(exclude_titles) + [limit])

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
        return [row['title'] for row in cursor.fetchall()]

    def exists(self, title: str) -> bool:
        """Check if an article exists."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT 1 FROM articles WHERE title = ? COLLATE NOCASE LIMIT 1", (title,))
        return cursor.fetchone() is not None


def _process_single_bz2_file(bz2_file: Path) -> Tuple[int, List["WikipediaArticle"], Optional[str]]:
    """
    Process a single BZ2 file and return (article_count, articles_list, error_message).

    Args:
        bz2_file: Path to the BZ2 file to process

    Returns:
        Tuple of (number of articles processed, list of articles, error message if any)
    """
    from .model import TokenCounter, WikipediaArticle
    articles = []
    error_message = None

    WikipediaArticle.token_counter = TokenCounter(method="gpt")

    try:
        with bz2.open(bz2_file, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.strip():
                    try:
                        article = WikipediaArticle.from_json(line)
                        articles.append(article)
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON lines
    except Exception as e:
        error_message = str(e)

    return len(articles), articles, error_message


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
    if dump_path.suffix == '.bz2' and tarfile.is_tarfile(dump_path):
        logger.info("Detected tar.bz2 file, extracting to temporary directory...")
        temp_extract_dir = Path(tempfile.mkdtemp())
        with tarfile.open(dump_path, 'r:bz2') as tar:
            tar.extractall(temp_extract_dir)
        base_path = temp_extract_dir
    elif dump_path.is_dir():
        logger.info("Detected extracted directory, using directly...")
    else:
        raise ValueError(f"Path {dump_path} is neither a valid tar.bz2 file nor a directory")

    # Find all .bz2 files in the base directory
    bz2_files = list(base_path.rglob('*.bz2'))
    bz2_files = [f for f in bz2_files if f.is_file()]

    logger.info(f"Found {len(bz2_files)} bz2 files to process")

    if not bz2_files:
        raise ValueError(f"No bz2 files found in {base_path}")

    total_articles = 0
    error_count = 0

    # Determine optimal number of processes
    max_processes = min(multiprocessing.cpu_count(), len(bz2_files))
    logger.info(f"Using {max_processes} parallel processes for loading")

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(_process_single_bz2_file, bz2_file): bz2_file for bz2_file in bz2_files}

        # Process results as they complete
        completed_files = 0
        with tqdm(total=len(bz2_files), desc="Ingesting BZ2 files to SQLite") as pbar:
            for future in as_completed(future_to_file):
                bz2_file = future_to_file[future]
                try:
                    article_count, articles, error_message = future.result()

                    if error_message:
                        logger.warning(f"Error processing {bz2_file.name}: {error_message}")
                        error_count += 1
                    else:
                        # Insert articles in batches
                        if articles:
                            # Batch the articles to avoid large transactions
                            chunk_size = batch_size
                            for i in range(0, len(articles), chunk_size):
                                chunk = articles[i:i + chunk_size]
                                index.bulk_insert(chunk)
                            total_articles += article_count

                except Exception as e:
                    logger.error(f"Failed to process {bz2_file.name}: {e}")
                    error_count += 1

                completed_files += 1
                pbar.update(1)

    # Cleanup temporary directory if used
    if temp_extract_dir:
        shutil.rmtree(temp_extract_dir)

    logger.info(f"Ingestion complete! Total articles inserted: {total_articles}, errors: {error_count}")

    if error_count > 0:
        logger.warning(f"There were {error_count} errors during processing")
