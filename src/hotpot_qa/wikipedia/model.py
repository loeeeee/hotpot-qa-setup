from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Self, Tuple, TypedDict, Union


@dataclass
class WikipediaArticle:
    pass

@dataclass
class Wikipedia:

    @classmethod
    def from_bz(cls, path: Path) -> Self:
        pass

@dataclass
class Tokenizer:
    _tokenizer: Literal["simple", "nltk"]

    def tokenize(self, query: str) -> List[str]:
        """
        main function of the tokenizer
        """
        # Choose tokenizer function based on settings
        pass

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
        pass
