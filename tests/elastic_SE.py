import unittest
from unittest.mock import MagicMock, patch
from src.core.search_engine import ElasticSearchEngine
from src.core.document import Document
from src.core.filter import Filter
import yaml


class DummyField:
    def __init__(self, name, type_value, max_len):
        self.name = name
        self.type = type("Type", (), {"value": type_value})()
        self.max_len = max_len

class DummyDocument(Document):
    @staticmethod
    def metadata_schema():
        return {
            "field1": DummyField("field1", "str", 100)
        }


class DummyFilter(Filter):
    @staticmethod
    def filter_fields():
        return ["field1"]

    @staticmethod
    def must_fields():
        return ["field1"]


class TestElasticSearchEngine(unittest.TestCase):
    @patch("src.core.search_engine.yaml.safe_load")
    @patch("src.core.search_engine.Elasticsearch")
    def setUp(self, mock_es, mock_yaml):
        # Mock elasticsearch.yml config
        mock_yaml.return_value = {
            "password": "test_password",
            "ca_certs": "/path/to/ca.crt"
        }

        # Mock Elasticsearch instance
        self.mock_es_instance = MagicMock()
        mock_es.return_value = self.mock_es_instance

        # Create ElasticSearchEngine instance
        self.engine = ElasticSearchEngine(
            es_host="http://localhost:9200",
            document_cls=DummyDocument,
            filter_cls=DummyFilter,
            es_index="test_index",
            force_rebuild=True
        )

    @patch("src.core.search_engine.ElasticIndexBuilder")
    def test_setup_calls_builder_build(self, mock_builder):
        self.engine.setup()
        mock_builder.assert_called_once()
        mock_builder.return_value.build.assert_called_once_with(force_rebuild=True)

    def test_insert_skips_existing_docs(self):
        # Mock existing document
        self.mock_es_instance.exists.return_value = True
        doc = MagicMock(spec=Document)
        doc.key.return_value = "doc1"
        doc.metadata.return_value = {"field1": MagicMock(contents="value")}
        self.engine.insert([doc])
        self.mock_es_instance.index.assert_not_called()

    def test_insert_inserts_new_docs(self):
        # Mock non-existing document
        self.mock_es_instance.exists.return_value = False
        doc = MagicMock(spec=Document)
        doc.key.return_value = "doc2"
        doc.metadata.return_value = {"field1": MagicMock(contents="value")}
        self.engine.insert([doc])
        self.mock_es_instance.index.assert_called_once()

    def test_search_builds_query_and_returns_ids(self):
        dummy_hits = [
            {"_id": "doc1"},
            {"_id": "doc2"}
        ]
        self.mock_es_instance.search.return_value = {"hits": {"hits": dummy_hits}}
        filter_obj = DummyFilter()
        result_ids = self.engine.search("query", filter_obj, limit=2)
        self.mock_es_instance.search.assert_called_once()
        self.assertEqual(result_ids, ["doc1", "doc2"])

    def test_spec_returns_correct_value(self):
        spec = self.engine.spec()
        self.assertEqual(spec.name, "elastic_search_engine")
        self.assertEqual(spec.optimal_for, "strong")


if __name__ == "__main__":
    unittest.main()
