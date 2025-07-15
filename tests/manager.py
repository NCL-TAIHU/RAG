import unittest
from unittest.mock import MagicMock, call
from src.core.manager import Manager

class TestManager(unittest.TestCase):
    def setUp(self):
        # Mock Library
        self.library = MagicMock()
        # Mock SearchEngine
        self.engine1 = MagicMock()
        self.engine2 = MagicMock()
        self.engine1.spec.return_value = MagicMock()
        self.engine2.spec.return_value = MagicMock()
        self.search_engines = [self.engine1, self.engine2]
        # Mock Reranker
        self.reranker = MagicMock()
        # Mock Router: use 'simple' so always index 0
        self.manager = Manager(self.library, self.search_engines, self.reranker, router_name="simple")
        # Mock Filter and Document
        self.filter = MagicMock()
        self.docs = [MagicMock(), MagicMock()]
        self.ids = ["id1", "id2"]

    def test_fetch_routes_and_reranks(self):
        # Setup engine and library mocks
        self.engine1.search.return_value = self.ids
        self.library.retrieve.return_value = self.docs
        self.reranker.rerank.return_value = list(reversed(self.docs))
        result = self.manager.fetch("query", self.filter, 2)
        self.engine1.search.assert_called_once_with("query", self.filter, 2)
        self.library.retrieve.assert_called_once_with(self.ids)
        self.reranker.rerank.assert_called_once_with("query", self.docs)
        self.assertEqual(result, list(reversed(self.docs)))

    def test_insert_calls_all(self):
        self.manager.insert(self.docs)
        self.library.insert.assert_called_once_with(self.docs)
        self.engine1.insert.assert_called_once_with(self.docs)
        self.engine2.insert.assert_called_once_with(self.docs)

    def test_setup_clears_and_setups(self):
        self.manager.setup()
        self.library.clear.assert_called_once()
        self.engine1.setup.assert_called_once()
        self.engine2.setup.assert_called_once()

if __name__ == "__main__":
    unittest.main() 