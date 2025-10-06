from dataclasses import dataclass, field
from .model import WikipediaArticle, Wikipedia, Tokenizer
from pathlib import Path
from typing import List, Self, Dict, Tuple
import pickle
import math
from collections import defaultdict
import logging
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _process_articles_batch(batch: List[Tuple[str, WikipediaArticle]], tokenizer_type: str) -> Tuple[Dict[str, int], Dict[str, List[Tuple[str, float]]]]:
    """
    Process a batch of articles and return local term document frequency and inverted index part.

    Args:
        batch: List of (title, article) tuples
        tokenizer_type: Tokenizer type ("simple" or "nltk")

    Returns:
        Tuple of (term_doc_freq_dict, inverted_index_part_dict)
    """
    from typing import Literal
    # Create tokenizer in the worker process
    tokenizer = Tokenizer(_tokenizer=tokenizer_type if tokenizer_type in ["simple", "nltk"] else "simple")  # type: ignore

    # Handle NLTK tokenizer initialization in worker process
    if tokenizer_type == "nltk":
        try:
            import nltk
            # Try to use punkt tokenizer to check if data is available
            nltk.word_tokenize("test")
        except LookupError:
            # Download punkt data if not available
            logger.info("Downloading NLTK punkt data in worker process...")
            nltk.download('punkt', quiet=True)

    local_term_doc_freq = defaultdict(int)
    local_inverted_index = defaultdict(list)

    for title, article in batch:
        # Tokenize article content (title + text)
        doc_tokens = tokenizer.tokenize((article.title + " " + article.text).lower())
        doc_length = len(doc_tokens)

        if doc_length == 0:
            continue

        token_counts = defaultdict(int)
        seen_terms = set()

        for token in doc_tokens:
            token_counts[token] += 1
            seen_terms.add(token)

        # Update local term-doc frequencies
        for term in seen_terms:
            local_term_doc_freq[term] += 1

        # Build local inverted index parts directly
        for term, count in token_counts.items():
            tf = count / doc_length
            local_inverted_index[term].append((title, tf))

    return dict(local_term_doc_freq), dict(local_inverted_index)


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
        construct a TF-IDF search index using parallel processing
        """

        logger.info("Building TF-IDF search index...")

        N = len(self.wikipedia.articles)

        # Determine batch size and number of processes
        num_processes = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid memory issues
        batch_size = max(1, N // (num_processes * 4))  # Aim for several batches per process

        # Split articles into batches
        articles_items = list(self.wikipedia.articles.items())
        batches = [articles_items[i:i + batch_size] for i in range(0, len(articles_items), batch_size)]

        logger.info(f"Processing {N} articles in {len(batches)} batches using {num_processes} processes")

        # Process batches in parallel
        global_term_doc_freq = defaultdict(int)
        global_inverted_index = defaultdict(list)

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(_process_articles_batch, batch, self.tokenizer._tokenizer): batch
                for batch in batches
            }

            # Collect results as they complete
            for future in tqdm(as_completed(future_to_batch), total=len(batches), desc="Processing batches"):
                try:
                    local_term_doc_freq, local_inverted_index = future.result()

                    # Merge term document frequencies
                    for term, freq in local_term_doc_freq.items():
                        global_term_doc_freq[term] += freq

                    # Merge inverted index parts
                    for term, entries in local_inverted_index.items():
                        global_inverted_index[term].extend(entries)

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    raise

        # Compute IDF
        self.idf_dict = {term: math.log(N / df) if df > 0 else 0 for term, df in global_term_doc_freq.items()}

        # Convert defaultdict to regular dict for inverted_index
        self.inverted_index = dict(global_inverted_index)

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

    def search_fuzzy_random(self, query: str) -> List[WikipediaArticle]:
        """
        get random articles from the Wikipedia
        """
        pass

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
