from collections.abc import Iterator
from dataclasses import dataclass
from .model import RawFullWikiQA, WikipediaArticle, Wikipedia, Tokenizer
from pathlib import Path

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
        pass

    def search_fuzzy(self, query: str) -> List[WikipediaArticle]:
        """
        search the entire article based on the input query
        """
        # Tokenize the query

        # Search

        # Rank

        pass
