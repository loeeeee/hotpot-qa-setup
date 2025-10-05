from collections.abc import Iterator
from dataclasses import dataclass
from .model import RawFullWikiQA, WikipediaArticle, Wikipedia, Tokenizer
from pathlib import Path

class RawFullWikiQAReader(Iterator):
    def __init__(self, full_wiki_path: Path) -> None:
        super().__init__()

    def __iter__(self) -> Iterator[_T_co]:
        return super().__iter__()

    def __next__(self) -> RawFullWikiQA:
        """
        return an instance of RawFullWikiQA from the loaded dictionary
        """

        pass

