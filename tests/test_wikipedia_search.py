import unittest
from pathlib import Path
from hotpot_qa.wikipedia.model import WikipediaArticle, Wikipedia, Tokenizer
from hotpot_qa.wikipedia.search import WikipediaSearchEngine


class TestWikipediaSearch(unittest.TestCase):

    def setUp(self):
        # Create mock articles
        article1 = WikipediaArticle(id="1", url="http://test.com/1", title="Test Article", text="This is a test article.")
        article2 = WikipediaArticle(id="2", url="http://test.com/2", title="Another Article", text="This is another article.")

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


if __name__ == '__main__':
    unittest.main()
