import unittest
from unittest.mock import patch, MagicMock, mock_open
from src.core.collection import (
    CollectionBuilder, CollectionConfig, FieldConfig, IndexConfig
)
from pymilvus import Collection, utility, DataType, connections


class TestCollectionBuilder(unittest.TestCase):
    connections.connect()

    def setUp(self):
        # ✅ Use pymilvus DataType enums (not strings)
        self.config = CollectionConfig(
            collection_name="test_collection",
            fields=[
                FieldConfig(name="pk", dtype=DataType.INT64, is_primary=True),
                FieldConfig(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=128)
            ],
            indexes=[
                IndexConfig(field_name="dense_vector", index_params={"index_type": "IVF_FLAT", "metric_type": "L2"})
            ],
            consistency_level="Strong"
        )
        self.builder = CollectionBuilder.from_config(self.config)

    @patch("src.core.collection.CollectionConfig")
    def test_from_config_creates_builder(self, mock_config):
        """from_config should create a CollectionBuilder instance"""
        mock_config.__dict__ = self.config.__dict__
        builder = CollectionBuilder.from_config(self.config)
        self.assertEqual(builder.collection_name, "test_collection")
        self.assertEqual(len(builder.fields), 2)
        self.assertEqual(len(builder.indexes), 1)

    def test_get_config_returns_correct_values(self):
        """get_config should return a CollectionConfig matching the builder"""
        returned_config = self.builder.get_config()
        self.assertEqual(returned_config.collection_name, "test_collection")
        self.assertEqual(returned_config.fields[0].name, "pk")

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save_and_load_config(self, mock_makedirs, mock_file):
        """_save_config should write JSON, _config_matches_saved should validate"""
        self.builder._save_config()
        mock_file.assert_called_with(self.builder.config_path, "w", encoding="utf-8")
        # Simulate loading same config for match check
        with patch("os.path.exists", return_value=True), \
             patch("json.load", return_value=self.config.model_dump()):
            result = self.builder._config_matches_saved()
            self.assertTrue(result)

    @patch("src.core.collection.utility.has_collection", return_value=True)
    @patch("src.core.collection.CollectionSchema")
    @patch("src.core.collection.FieldSchema")
    @patch("src.core.collection.Collection")
    def test_build_creates_collection(
        self, mock_collection, mock_field_schema, mock_schema, mock_has_collection
    ):
        """build should call Milvus APIs and return Collection"""
        # Create a mock Collection instance
        mock_instance = MagicMock()
        mock_collection.return_value = mock_instance

        # Hook .drop and .create_index on the mock Collection instance
        mock_instance.drop = MagicMock()
        mock_instance.create_index = MagicMock()

        result = self.builder.build()

        # ✅ Assertions
        mock_has_collection.assert_called_once_with("test_collection")
        mock_instance.drop.assert_called_once()
        mock_instance.create_index.assert_called()  # ✅ Fix
        self.assertEqual(result, mock_instance)

    @patch("src.core.collection.os.path.exists", return_value=True)
    @patch("src.core.collection.json.load")
    @patch("src.core.collection.Collection")
    def test_get_existing_returns_collection(self, mock_collection, mock_json_load, mock_exists):
        """get_existing should return Collection if config matches"""
        mock_json_load.return_value = self.config.model_dump()
        result = self.builder.get_existing()
        self.assertEqual(result, mock_collection.return_value)

    @patch("src.core.collection.os.path.exists", return_value=True)
    @patch("src.core.collection.json.load")
    def test_get_existing_returns_none_if_mismatch(self, mock_json_load, mock_exists):
        """get_existing should return None if config does not match"""
        mock_json_load.return_value = {**self.config.model_dump(), "collection_name": "wrong_name"}
        result = self.builder.get_existing()
        self.assertIsNone(result)

    def check_the_collection_content(self):
        """check the collection content"""
        collection = self.builder.build()
        print(collection.num_entities)
        print(collection.schema)
        print(collection.load())
        print(collection.query("", limit=10))

    def test_check_the_collection_content(self):
        """check the collection content"""
        self.check_the_collection_content()


if __name__ == "__main__":
    unittest.main(verbosity=2)
