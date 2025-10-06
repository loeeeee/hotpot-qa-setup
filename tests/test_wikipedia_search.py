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

    def test_pickle_roundtrip(self):
        # Test save and load functionality
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save to pickle
            self.search_engine.to_pickle(tmp_path)

            # Load from pickle
            loaded_engine = WikipediaSearchEngine.from_pickle(tmp_path)

            # Verify the loaded engine has the same data
            self.assertEqual(len(loaded_engine.wikipedia.articles), len(self.search_engine.wikipedia.articles))
            self.assertEqual(list(loaded_engine.wikipedia.articles.keys()), list(self.search_engine.wikipedia.articles.keys()))

            # Test that search still works
            results = loaded_engine.search_title("Test Article")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "Test Article")

        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
