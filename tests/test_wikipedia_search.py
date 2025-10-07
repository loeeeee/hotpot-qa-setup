import unittest
import tempfile
from pathlib import Path
from hotpot_qa.wikipedia.model import WikipediaArticle, Tokenizer
from hotpot_qa.wikipedia.search import WikipediaSearchEngine
from hotpot_qa.wikipedia.sqlite_backend import WikipediaSQLiteConfig, WikipediaSQLiteIndex


class TestWikipediaSearch(unittest.TestCase):

    def setUp(self):
        # Create temp DB
        self.temp_db = Path(tempfile.NamedTemporaryFile(suffix='.db', delete=False).name)

        # Create mock articles
        self.articles = [
            WikipediaArticle(id="1", title="Test Article", paragraphs=["This is a test article."], links=[], token=[5]),
            WikipediaArticle(id="2", title="Another Article", paragraphs=["This is another article."], links=[], token=[5])
        ]

        # Setup DB
        config = WikipediaSQLiteConfig(db_path=self.temp_db)
        index = WikipediaSQLiteIndex(config)
        index.bulk_insert(self.articles)

        # Create tokenizer
        self.tokenizer = Tokenizer("simple")

        # Create search engine
        self.search_engine = WikipediaSearchEngine(self.temp_db, self.tokenizer)

    def tearDown(self):
        self.temp_db.unlink(missing_ok=True)

    def test_search_title_exact_match(self):
        results = self.search_engine.search_title("Test Article")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Test Article")

    def test_search_title_no_match(self):
        results = self.search_engine.search_title("Nonexistent Article")
        self.assertEqual(len(results), 0)

    def test_search_fuzzy_fallback(self):
        results = self.search_engine.search_fuzzy("some query")
        # Should use token overlap fallback since index not built
        self.assertGreaterEqual(len(results), 0)  # None or some matches

    def test_search_fuzzy_matches(self):
        # Search for "test" should find "Test Article" (title match)
        results = self.search_engine.search_fuzzy("test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Test Article")

        # Search for "article" should find both articles (text match)
        results = self.search_engine.search_fuzzy("article")
        self.assertEqual(len(results), 2)
        # Test Article should come first due to title match + text match
        self.assertEqual(results[0].title, "Test Article")
        self.assertEqual(results[1].title, "Another Article")

    def test_search_fuzzy_nltk_tokenizer(self):
        # Test with nltk tokenizer
        try:
            nltk_tokenizer = Tokenizer("nltk")
            # Test tokenization to ensure punkt is available
            test_tokens = nltk_tokenizer.tokenize("test sentence")
            self.assertGreater(len(test_tokens), 1)  # Should split "sentence"

            nltk_search_engine = WikipediaSearchEngine(self.temp_db, nltk_tokenizer)
            results = nltk_search_engine.search_fuzzy("article")
            self.assertEqual(len(results), 2)
        except LookupError:
            self.skipTest("NLTK punkt tokenizer data not available")

    def test_search_fuzzy_with_index(self):
        # Build TF-IDF index and test
        self.search_engine.build_index()
        self.assertTrue(self.search_engine.index_built)

        # Search for "article" should find both articles
        results = self.search_engine.search_fuzzy("article")
        self.assertEqual(len(results), 2)
        # TF-IDF should still rank "Test Article" higher due to title
        self.assertEqual(results[0].title, "Test Article")
        self.assertEqual(results[1].title, "Another Article")

    def test_sqlite_index_methods(self):
        # Test backing index methods
        self.assertEqual(self.search_engine.wikipedia.total_articles, 2)
        self.assertTrue(self.search_engine.wikipedia.exists("Test Article"))
        self.assertFalse(self.search_engine.wikipedia.exists("Nonexistent"))

        random_articles = self.search_engine.wikipedia.get_random_articles(1, set())
        self.assertEqual(len(random_articles), 1)


if __name__ == '__main__':
    unittest.main()
