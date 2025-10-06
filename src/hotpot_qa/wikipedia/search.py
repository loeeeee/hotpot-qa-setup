from collections.abc import Iterator
from dataclasses import dataclass, field
from .model import WikipediaArticle, Wikipedia, Tokenizer
from pathlib import Path
from typing import List, Self, Dict, Tuple
import pickle
import math
from collections import defaultdict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class WikipediaSearchEngine:
    wikipedia: Wikipedia
    tokenizer: Tokenizer
    idf_dict: Dict[str, float] = field(default_factory=dict)
    inverted_index: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    index_built: bool = False

    # Internal
    def _build_search_index(self) -> None:
        """
        construct a TF-IDF search index
        """

        logger.info("Building TF-IDF search index...")

        N = len(self.wikipedia.articles)
        term_doc_freq = defaultdict(int)
        term_doc_tf = defaultdict(lambda: defaultdict(float))

        # First pass: collect term frequencies and document frequencies
        for title, article in tqdm(self.wikipedia.articles.items(), desc="Processing articles", unit="article"):
            # Tokenize article content (title + text)
            doc_tokens = self.tokenizer.tokenize((article.title + " " + article.text).lower())
            doc_length = len(doc_tokens)

            if doc_length == 0:
                continue

            token_counts = defaultdict(int)
            seen_terms = set()

            for token in doc_tokens:
                token_counts[token] += 1
                seen_terms.add(token)

            # Update term-doc frequencies
            for term in seen_terms:
                term_doc_freq[term] += 1

            # Store TF for this document
            for term, count in token_counts.items():
                term_doc_tf[term][title] = count / doc_length

        # Compute IDF
        self.idf_dict = {term: math.log(N / df) if df > 0 else 0 for term, df in term_doc_freq.items()}

        # Build inverted index
        self.inverted_index = defaultdict(list)
        for term, docs in term_doc_tf.items():
            for doc_id, tf in docs.items():
                self.inverted_index[term].append((doc_id, tf))

        self.index_built = True
        logger.info(f"Built TF-IDF index with {len(self.idf_dict)} terms across {N} documents")

    def _build_page_rank(self) -> None:
        """
        construct a page rank search index based on hyperlinks
        """
        pass

    # API
    def build_index(self) -> None:
        """
        Build the TF-IDF search index for fuzzy search
        """
        self._build_search_index()

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
        from collections import defaultdict

        query_tokens = self.tokenizer.tokenize(query.lower())

        if not query_tokens:
            return []

        if self.index_built and self.idf_dict and self.inverted_index:
            # Use TF-IDF scoring
            scores = defaultdict(float)
            query_length = len(query_tokens)
            unique_query_tokens = set(query_tokens)

            for token in unique_query_tokens:
                if token not in self.idf_dict:
                    continue
                idf = self.idf_dict[token]
                tf_query = query_tokens.count(token) / query_length
                query_weight = tf_query * idf

                for doc_id, tf_doc in self.inverted_index.get(token, []):
                    scores[doc_id] += query_weight * (tf_doc * idf)

            # Sort by score descending
            results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [self.wikipedia.articles[doc_id] for doc_id, _ in results]
        else:
            # Fallback to token overlap
            query_token_set = set(query_tokens)
            results = []

            for article in self.wikipedia.articles.values():
                # Tokenize title and text (lowercased)
                title_tokens = set(self.tokenizer.tokenize(article.title.lower()))
                text_tokens = set(self.tokenizer.tokenize(article.text.lower()))

                # Calculate matches
                title_matches = len(query_token_set & title_tokens)
                text_matches = len(query_token_set & text_tokens)

                # Score: title matches weighted 10x higher than text matches
                score = (title_matches * 10) + text_matches

                if score > 0:
                    results.append((article, score))

            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)

            # Return articles only
            return [article for article, score in results]

    # Cache System
    @classmethod
    def from_pickle(cls, path: Path) -> Self:
        """
        restore cache from path
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def to_pickle(self, path: Path) -> None:
        """
        pickle all
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
