import unittest
import tempfile
from pathlib import Path
from hotpot_qa.wikipedia.model import WikipediaArticle, Wikipedia, Tokenizer
from hotpot_qa.wikipedia.search import WikipediaSearchEngine


class TestWikipediaSearch(unittest.TestCase):

    def setUp(self):
        # Create mock articles
        article1 = WikipediaArticle(id="1", title="Test Article", paragraphs=["This is a test article."], links=[], token=[5])
        article2 = WikipediaArticle(id="2", title="Another Article", paragraphs=["This is another article."], links=[], token=[5])

        # Create mock Wikipedia instance
        wiki = Wikipedia()
        wiki.articles["Test Article"] = article1
        wiki.articles["Another Article"] = article2

        # Create tokenizer
        tokenizer = Tokenizer("simple")

        # Create search engine
        self.search_engine = WikipediaSearchEngine(wiki, tokenizer)

    def test_search_title_exact_match(self):
        results = self.search_engine.search_title("Test Article")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Test Article")

    def test_search_title_no_match(self):
        results = self.search_engine.search_title("Nonexistent Article")
        self.assertEqual(len(results), 0)

    def test_search_fuzzy_placeholder(self):
        results = self.search_engine.search_fuzzy("some query")
        self.assertEqual(len(results), 0)

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

            nltk_search_engine = WikipediaSearchEngine(self.search_engine.wikipedia, nltk_tokenizer)
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

    def test_pickle_roundtrip(self):
        # Test save and load functionality
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Build TF-IDF index before pickling
            self.search_engine.build_index()

            # Save to pickle
            self.search_engine.to_pickle(tmp_path)

            # Load from pickle
            loaded_engine = WikipediaSearchEngine.from_pickle(tmp_path)

            # Verify the loaded engine has the same data
            self.assertEqual(len(loaded_engine.wikipedia.articles), len(self.search_engine.wikipedia.articles))
            self.assertEqual(list(loaded_engine.wikipedia.articles.keys()), list(self.search_engine.wikipedia.articles.keys()))

            # Verify TF-IDF data is preserved
            self.assertTrue(loaded_engine.index_built)
            self.assertEqual(len(loaded_engine.idf_dict), len(self.search_engine.idf_dict))
            self.assertEqual(len(loaded_engine.inverted_index), len(self.search_engine.inverted_index))

            # Test that search still works
            results = loaded_engine.search_title("Test Article")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "Test Article")

            # Test fuzzy search with TF-IDF
            fuzzy_results = loaded_engine.search_fuzzy("article")
            self.assertEqual(len(fuzzy_results), 2)

        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
