from collections.abc import Iterator
from dataclasses import dataclass
from .model import WikipediaArticle, Wikipedia, Tokenizer
from pathlib import Path
from typing import List

@dataclass
class WikipediaSearchEngine:
    wikipedia: Wikipedia
    tokenizer: Tokenizer

    # Internal
    def _build_search_index(self) -> None:
        """
        construct a TF-IDF search index
        """
        pass

    def _build_page_rank(self) -> None:
        """
        construct a page rank search index based on hyperlinks
        """
        pass

    # API
    def search_title(self, query: str) -> List[WikipediaArticle]:
        """
        search the title and find the corresponding article
        """
        if query in self.wikipedia.articles:
            return [self.wikipedia.articles[query]]
        return []

    def search_fuzzy(self, query: str) -> List[WikipediaArticle]:
        """
        search the entire article based on the input query
        """
        # Tokenize the query

        # Search

        # Rank
        return []

    # Cache System
    @classmethod
    def from_pickle(cls, path: Path) -> Self:
        """
        restore cache from path
        """
        pass

    def to_pickle(self, path: Path) -> None:
        """
        pickle self.wikipedia, self.tokenizer
        """
        pass
