import unittest
from unittest.mock import MagicMock, patch
from src.core.search_engine import MilvusSearchEngine
from src.core.vector_manager import VectorManager
from src.core.document import Document
from src.core.filter import Filter
from src.core.collection import CollectionOperator
from src.core.collection import CollectionBuilder
from pymilvus import DataType  # <-- Import valid Milvus data types

# Add a dummy field class for schema
class DummyField:
    def __init__(self, name, dtype, max_len):
        self.name = name
        self.dtype = dtype
        self.max_len = max_len
    @property
    def type(self):
        return self
    def to_milvus_type(self):
        return self.dtype  # Return the int code expected by your system

class DummyDocument(Document):
    @staticmethod
    def metadata_schema():
        # Use a valid int for dtype (e.g., DataType.INT64 for INT64)
        return {
            "field1": DummyField(name="field1", dtype=DataType.INT64, max_len=100)
        }

class DummyFilter(Filter):
    @staticmethod
    def filter_fields():
        return ["field1"]

class TestMilvusSearchEngine(unittest.TestCase):
    def setUp(self):
        # Mock VectorManager
        self.mock_vm = MagicMock(spec=VectorManager)
        vs_meta = MagicMock()
        vs_meta.model = "test_model"
        vs_meta.dataset = "test_dataset"
        vs_meta.channel = "abstract"
        vs_meta.chunker_meta.chunker_type = "length"
        self.mock_vm.get_vs_metadata.return_value = vs_meta
        self.mock_vm.get_dim = MagicMock(return_value=128)
        self.mock_vm.embedder = MagicMock()
        # Always return a single vector in a list for dense embedding
        self.mock_vm.embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        self.mock_vm.embedder.get_dim.return_value = 128

        self.engine = MilvusSearchEngine(
            vector_type="dense",
            vector_manager=self.mock_vm,
            document_cls=DummyDocument,
            filter_cls=DummyFilter,
            force_rebuild=True
        )

    # Patch where the classes are used in src.core.search_engine, not where defined
    @patch("src.core.search_engine.CollectionBuilder")
    @patch("src.core.search_engine.CollectionOperator")
    def test_setup_calls_builder_and_operator(self, mock_operator, mock_builder):
        # Setup the builder and operator mocks
        mock_builder.from_config.return_value.connect.return_value = None
        mock_builder.from_config.return_value.build.return_value = MagicMock(name="Collection")
        self.engine.setup()
        # Debug: print if from_config was called
        print(f"from_config call count: {mock_builder.from_config.call_count}")
        mock_builder.from_config.assert_called_once()
        mock_operator.assert_called_once()

    @patch("src.core.search_engine.CollectionOperator")
    def test_insert_skips_existing_docs(self, mock_operator):
        mock_operator.return_value.query.return_value = [{"pk": "doc1-0"}]
        self.engine.operator = mock_operator.return_value
        mock_docs = [MagicMock(spec=Document)]
        mock_docs[0].key.return_value = "doc1"
        self.engine.setup()  # Ensure collection is set up before insert
        self.engine.insert(mock_docs)
        # Assert that buffered_insert is called once with empty data (as observed)
        mock_operator.return_value.buffered_insert.assert_called_once_with([[], [], []])

    @patch("src.core.search_engine.CollectionOperator")
    def test_search_calls_operator_search(self, mock_operator):
        self.engine.operator = mock_operator.return_value
        mock_operator.return_value.search.return_value = [[MagicMock(fields={"pk": "doc1-0"})]]
        # Mock get_raw_embedding to return a list of length 1 (as expected by embed_query)
        self.mock_vm.get_raw_embedding.return_value = [[0.1, 0.2, 0.3]]
        results = self.engine.search("query", MagicMock(spec=Filter), limit=5)
        mock_operator.return_value.search.assert_called_once()
        self.assertIsInstance(results, list)

    def test_insert_with_duplicate_docs(self):
        self.engine.setup()
        bad_doc = MagicMock(spec=Document)
        bad_doc.key.return_value = "doc1"
        with self.assertRaises(ValueError):
            self.engine.insert([bad_doc, bad_doc])

    def test_insert_with_empty_docs(self):
        self.engine.setup()
        with self.assertRaises(ValueError):
            self.engine.insert([])
    
    def test_insert_with_empty_key(self):
        self.engine.setup()
        bad_doc = MagicMock(spec=Document)
        bad_doc.key.return_value = ""
        with self.assertRaises(ValueError):
            self.engine.insert([bad_doc])

if __name__ == "__main__":
    unittest.main()
