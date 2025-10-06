import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Self, Optional, Tuple
import bz2
import json
import tarfile
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re
import urllib.parse as urlparse
import tiktoken
import nltk


@dataclass
class TokenCounter:
    method: Literal["simple", "gpt"]

    def count_token(self, text: str) -> int:
        """
        main interface of the TokenCounter, route to specific method
        """
        if self.method == "simple":
            return TokenCounter._simple(text)
        elif self.method == "gpt":
            return TokenCounter._gpt(text)
        else:
            raise ValueError("Unknown method")

    @staticmethod
    def _simple(text: str) -> int:
        """
        count words based on the space
        """
        return len(text.split())

    @staticmethod
    def _gpt(text: str) -> int:
        """
        count words based on gpt tokenizer
        """
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))


@dataclass
class WikipediaArticle:
    id: str             # Unique ID loaded from the wikipedia dump
    title: str          # A unique string loaded from the wikipedia dump, the primary way of referring to data
    paragraphs: List[str] = field(default_factory=list)           # The text without links to other articles
    token: List[int] = field(default_factory=list)                # The number of tokens for each paragraph
    links: List[str] = field(default_factory=list)                # A collection of all the links to the other wikipedia article

    # Classvar
    token_counter: ClassVar[TokenCounter]

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)

    @staticmethod
    def _clean_and_extract(text: str):
        links = set()
        pattern = re.compile(r'<a href="([^"]+)">(.*?)</a>', re.IGNORECASE)
        def replacer(match):
            link = urlparse.unquote(match.group(1))
            display = match.group(2)
            links.add(link)
            return display
        clean_text = pattern.sub(replacer, text)
        return clean_text, links

    @classmethod
    def from_json(cls, line: str) -> Self:
        """
        load an instance of Wikipedia Article from a string that is supposed have JSON format
        """

        data = json.loads(line)
        id = data['id']
        title = data['title']

        paragraphs = []
        all_links = set()
        for para_sentences in data['text']:
            if not para_sentences:
                continue
            full_para = ' '.join(para_sentences)
            clean_para, para_links = cls._clean_and_extract(full_para)
            paragraphs.append(clean_para)
            all_links.update(para_links)

        links = list(all_links)
        token_counts = [cls.token_counter.count_token(p) for p in paragraphs]

        return cls(id=id, title=title, paragraphs=paragraphs, links=links, token=token_counts)


def process_single_bz2_file(bz2_file: Path) -> Tuple[int, Dict[str, WikipediaArticle], Optional[str]]:
    """
    Process a single BZ2 file and return (article_count, articles_dict, error_message).

    Args:
        bz2_file: Path to the BZ2 file to process
        token_counter: TokenCounter instance

    Returns:
        Tuple of (number of articles processed, dict of articles by title, error message if any)
    """
    articles = {}
    article_count = 0
    error_message = None

    WikipediaArticle.token_counter = TokenCounter(method="gpt")

    try:
        with bz2.open(bz2_file, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    try:
                        article = WikipediaArticle.from_json(line)
                        articles[article.title] = article
                        article_count += 1
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON lines
    except Exception as e:
        error_message = str(e)
        article_count = 0
        articles = {}

    return article_count, articles, error_message



@dataclass
class Wikipedia:
    articles: Dict[str, WikipediaArticle] = field(default_factory=dict)
    token_counter: TokenCounter = field(default_factory=lambda: TokenCounter("simple"))

    @classmethod
    def from_bz(cls, path: Path) -> Self:
        """
        Load Wikipedia articles from either:
        1. A tar.bz2 file containing multiple bz2 files with JSON lines
        2. A directory containing extracted bz2 files
        """
        instance = cls()
        logger = logging.getLogger(__name__)

        base_path = path

        # Handle tar.bz2 file - extract to temporary directory
        if path.suffix == '.bz2' and tarfile.is_tarfile(path):
            logger.info("Detected tar.bz2 file, extracting to temporary directory...")
            temp_extract_dir = Path(tempfile.mkdtemp())
            with tarfile.open(path, 'r:bz2') as tar:
                tar.extractall(temp_extract_dir)
            base_path = temp_extract_dir
        elif path.is_dir():
            logger.info("Detected extracted directory, using directly...")
        else:
            raise ValueError(f"Path {path} is neither a valid tar.bz2 file nor a directory")

        # Find all .bz2 files in the base directory
        bz2_files = []
        for file_path in base_path.rglob('*.bz2'):
            bz2_files.append(file_path)

        logger.info(f"Found {len(bz2_files)} bz2 files to process")

        total_articles = 0

        # Determine optimal number of processes
        max_processes = min(multiprocessing.cpu_count(), len(bz2_files))  # Cap at 8 to avoid memory issues
        logger.info(f"Using {max_processes} parallel processes for loading")

        # Try parallel processing
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_single_bz2_file, bz2_file): bz2_file for bz2_file in bz2_files}

            # Process results as they complete
            completed_files = 0
            with tqdm(total=len(bz2_files), desc="Processing BZ2 files") as pbar:
                for future in as_completed(future_to_file):
                    bz2_file = future_to_file[future]
                    try:
                        file_articles, file_articles_dict, error_message = future.result()

                        if error_message:
                            logger.warning(f"Error processing {bz2_file.name}: {error_message}")
                        else:
                            # Merge results into main articles dict
                            collision_count = 0
                            for title, article in file_articles_dict.items():
                                if title in instance.articles:
                                    logger.warning(f"Title collision for '{title}' from {bz2_file.name}, overwriting")
                                    collision_count += 1
                                instance.articles[title] = article

                            total_articles += file_articles
                            # tqdm.write(f"Processed {bz2_file.name}: {file_articles} articles ({collision_count} collisions)")

                    except Exception as e:
                        logger.warning(f"Failed to get result for {bz2_file.name}: {e}")

                    completed_files += 1
                    pbar.update(1)

        logger.info(f"Total articles loaded: {total_articles}")
        return instance


@dataclass
class Tokenizer:
    _tokenizer: Literal["simple", "nltk"]

    def tokenize(self, query: str) -> List[str]:
        """
        main function of the tokenizer
        """
        # Choose tokenizer function based on settings
        if self._tokenizer == "simple":
            return self._simple(query)
        elif self._tokenizer == "nltk":
            return self._nltk(query)
        else:
            raise ValueError(f"Unknown tokenizer: {self._tokenizer}")

    @staticmethod
    def _simple(query: str) -> List[str]:
        """
        Remove white space and keep case
        """
        return query.lstrip().split(" ")

    def _nltk(self, query: str) -> List[str]:
        """
        Use NLTK to tokenize the query
        """
        return nltk.word_tokenize(query)

