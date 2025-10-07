from dataclasses import dataclass, field
from .model import WikipediaArticle, Wikipedia, Tokenizer
from pathlib import Path
from typing import List, Self, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

logger = logging.getLogger(__name__)



@dataclass
class WikipediaSearchEngine:
    db_path: Path
    tokenizer: Tokenizer
    wikipedia: Wikipedia = field(init=False)
    vectorizer: Optional[TfidfVectorizer] = field(default=None)
    tfidf_matrix: Optional[object] = field(default=None)  # scipy sparse matrix
    index_built: bool = False

    def __post_init__(self):
        self.wikipedia = Wikipedia(db_path=self.db_path)

    # Internal
    def _build_search_index(self) -> None:
        """
        construct a TF-IDF search index using sklearn TfidfVectorizer
        """

        logger.info("Building TF-IDF search index using sklearn TfidfVectorizer...")

        N = self.wikipedia.total_articles

        # Prepare documents: (title + " " + text).lower()
        documents = []

        titles = self.wikipedia.index.get_all_titles()
        for title in tqdm(titles, desc="Loading documents for TF-IDF"):
            article = self.wikipedia.get_article_by_title(title)
            if article:
                documents.append((article.title + " " + article.text).lower())

        N_actual = len(documents)
        logger.info(f"Loaded {N_actual} documents out of {N} total articles")

        # Create TfidfVectorizer with custom tokenizer, no normalization for matching current scoring
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer.tokenize,
            lowercase=False,  # already lowercased in documents
            norm="l2",
            smooth_idf=False,  # IDF = log(n/df), close to current log(N/df)
            sublinear_tf=False,  # linear TF scaling
            use_idf=True
        )

        logger.info(f"Fitting and transforming {N_actual} documents...")
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        self.index_built = True
        logger.info(f"Built TF-IDF index with {len(self.vectorizer.vocabulary_)} terms across {N_actual} documents")

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
        article = self.wikipedia.get_article_by_title(query)
        return [article] if article else []

    def search_fuzzy_random(self, query: str) -> List[WikipediaArticle]:
        """
        get random articles from the Wikipedia
        """
        # TODO: implement random search
        return []

    def search_fuzzy(self, query: str) -> List[WikipediaArticle]:
        """
        search the entire article based on the input query
        """
        if not self.index_built or self.vectorizer is None or self.tfidf_matrix is None:
            # Fallback to token overlap
            query_tokens = self.tokenizer.tokenize(query.lower())
            if not query_tokens:
                return []

            query_token_set = set(query_tokens)
            results = []

            titles = self.wikipedia.index.get_all_titles()
            for title in titles:
                article = self.wikipedia.get_article_by_title(title)
                if article:
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

        # Use TF-IDF scoring with sklearn
        query_str = query.lower()
        query_vector = self.vectorizer.transform([query_str])

        # Compute dot product: query_vector @ tfidf_matrix.T
        scores = query_vector.dot(self.tfidf_matrix.T).toarray().flatten()

        # Get titles in same order as documents (already used in build)
        titles = self.wikipedia.index.get_all_titles()

        # Pair titles with scores and sort by score descending
        results = sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)

        # Return articles, filtering out zero scores
        return [article for title, score in results if score > 0 for article in [self.wikipedia.get_article_by_title(title)] if article is not None]
