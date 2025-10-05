from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Self
import bz2
import json


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
        Load Wikipedia articles from a bz2 file containing JSON lines.
        """
        instance = cls()
        with bz2.open(path, 'rt') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    id = data['id']
                    url = data['url']
                    title = data['title']
                    text = ' '.join([' '.join(paragraph) for paragraph in data['text']])
                    article = WikipediaArticle(id=id, url=url, title=title, text=text)
                    instance.articles[title] = article
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
