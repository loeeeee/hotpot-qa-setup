import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Literal, Self, Optional, Tuple, Set, Dict, cast
import bz2
import json
import multiprocessing
import re
import urllib.parse as urlparse
import tiktoken
import nltk

from .sqlite_backend import WikipediaSQLiteIndex, WikipediaSQLiteConfig


@dataclass
class TokenCounter:
    method: Literal["simple", "gpt"]
    _gpt_encoding: ClassVar[Optional[tiktoken.Encoding]] = None

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
        encoding = TokenCounter._gpt_encoding
        if encoding is None:
            encoding = tiktoken.get_encoding("cl100k_base")
            TokenCounter._gpt_encoding = encoding
        return len(encoding.encode(text))


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

    @property
    def total_token(self) -> int:
        return sum(self.token)

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


def process_single_bz2_file(bz2_file: Path) -> Tuple[int, List[WikipediaArticle], Optional[str]]:
    """
    Process a single BZ2 file and return (article_count, articles_list, error_message).

    Args:
        bz2_file: Path to the BZ2 file to process

    Returns:
        Tuple of (number of articles processed, list of articles, error message if any)
    """
    articles = []
    error_message = None

    WikipediaArticle.token_counter = TokenCounter(method="gpt")

    try:
        with bz2.open(bz2_file, 'rt', encoding='utf-8', errors='ignore') as f:
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



@dataclass
class Wikipedia:
    db_path: Path
    index: WikipediaSQLiteIndex = field(init=False)
    token_counter: TokenCounter = field(default_factory=lambda: TokenCounter("simple"))

    def __post_init__(self):
        config = WikipediaSQLiteConfig(db_path=self.db_path)
        self.index = WikipediaSQLiteIndex(config=config)

    def get_article_by_title(self, title: str) -> Optional[WikipediaArticle]:
        return self.index.get_article_by_title(title)

    def get_random_articles(self, limit: int, exclude_titles: Set[str]) -> List[WikipediaArticle]:
        return self.index.get_random_articles(limit, exclude_titles)

    def exists(self, title: str) -> bool:
        return self.index.exists(title)

    @property
    def total_articles(self) -> int:
        return self.index.get_article_count()


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
