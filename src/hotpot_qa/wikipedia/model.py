import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Self
import bz2
import json
import tarfile
import tempfile
from tqdm import tqdm


@dataclass
class WikipediaArticle:
    id: str
    url: str
    title: str
    text: str


@dataclass
class Wikipedia:
    articles: Dict[str, WikipediaArticle] = field(default_factory=dict)

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
        # Process each bz2 file
        for bz2_file in tqdm(bz2_files, desc="Processing BZ2 files"):
            file_articles = 0
            try:
                with bz2.open(bz2_file, 'rt', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                id = data['id']
                                url = data.get('url', f'https://en.wikipedia.org/wiki/{data["title"].replace(" ", "_")}')
                                title = data['title']
                                # Concatenate paragraphs
                                if isinstance(data['text'], list):
                                    if data['text'] and isinstance(data['text'][0], list):
                                        # List of paragraphs, where each paragraph is a list of sentences
                                        text = ' '.join([' '.join(para) for para in data['text']])
                                    else:
                                        # List of strings (sentences)
                                        text = ' '.join(data['text'])
                                else:
                                    text = str(data['text'])

                                article = WikipediaArticle(id=id, url=url, title=title, text=text)
                                instance.articles[title] = article
                                file_articles += 1
                            except json.JSONDecodeError:
                                continue  # Skip malformed JSON lines
                total_articles += file_articles
                tqdm.write(f"Processed {bz2_file.name}: {file_articles} articles")
            except Exception as e:
                logger.warning(f"Error processing {bz2_file}: {e}")
                continue

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
        try:
            import nltk
            return nltk.word_tokenize(query)
        except ImportError:
            raise ImportError("NLTK required for nltk tokenizer")


# Todo List (Optional - Plan Mode)
