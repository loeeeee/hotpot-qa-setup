import json
from collections.abc import Iterator
from typing import List
from .model import RawFullWikiQA
from pathlib import Path

class RawFullWikiQALoader(Iterator):
    def __init__(self, full_wiki_path: Path) -> None:
        with open(full_wiki_path, 'r', encoding='utf-8') as f:
            self.data: List[RawFullWikiQA] = json.load(f)
        self.index = 0

    def __iter__(self) -> Iterator[RawFullWikiQA]:
        return self

    def __next__(self) -> RawFullWikiQA:
        """
        return an instance of RawFullWikiQA from the loaded dictionary
        """
        if self.index >= len(self.data):
            raise StopIteration
        item = self.data[self.index]
        self.index += 1
        return item
