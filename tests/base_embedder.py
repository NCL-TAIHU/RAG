import unittest
import numpy as np
from scipy.sparse import csr_matrix
from src.core import embedder
from src.core.embedder import (
    BaseEmbedder, DenseEmbedder, SparseEmbedder,
    AutoModelEmbedder, BGEM3Embedder, MilvusBGEM3Embedder
)
from unittest.mock import patch


class TestEmbedder(unittest.TestCase):
    def setUp(self):
        """
        Set up common test data.
        """
        self.dummy_texts = ["拜拜", "祭祀", "在日治背景下，台灣社會如何因應外來的營養學理論與健康觀念？"]

    # ----------- 🔵 AutoModelEmbedder (Dense) -----------
    def test_dense_embedder_name_and_dim(self):
        """Check AutoModelEmbedder name and get_dim methods."""
        with patch.object(embedder, "AutoTokenizer"), \
             patch.object(embedder, "AutoModel"), \
             patch.object(embedder, "GPUtil", return_value=[0]):
            embedder_instance = AutoModelEmbedder("bert-base-uncased")
            embedder_instance.model.config.hidden_size = 768  # Mock hidden size
            self.assertEqual(embedder_instance.name(), "bert-base-uncased")
            self.assertEqual(embedder_instance.get_dim(), 768)

    def test_dense_embedder_embed_returns_list(self):
        """Test AutoModelEmbedder embed method produces dense vectors."""
        with patch.object(embedder, "AutoTokenizer"), \
             patch.object(embedder, "AutoModel"), \
             patch.object(embedder, "GPUtil", return_value=[0]), \
             patch.object(AutoModelEmbedder, "_embed_batch", return_value=[[0.1]*768]*3):
            embedder_instance = AutoModelEmbedder("bert-base-uncased")
            vectors = embedder_instance.embed(self.dummy_texts)
            self.assertEqual(len(vectors), 3)
            self.assertEqual(len(vectors[0]), 768)

    # ----------- 🟢 BGEM3Embedder (Sparse) -----------
    def test_sparse_embedder_name(self):
        """Check BGEM3Embedder name method."""
        with patch.object(embedder, "BGEM3FlagModel"), \
             patch.object(embedder, "GPUtil", return_value=[0]):
            embedder_instance = BGEM3Embedder("BAAI/bge-m3")
            self.assertEqual(embedder_instance.name(), "BAAI/bge-m3")

    def test_sparse_embedder_embed_returns_csr_array(self):
        """Test BGEM3Embedder embed method produces sparse matrix."""
        dummy_sparse = csr_matrix(np.eye(3))
        with patch.object(embedder, "BGEM3FlagModel"), \
             patch.object(BGEM3Embedder, "embed", return_value=dummy_sparse):
            embedder_instance = BGEM3Embedder("BAAI/bge-m3")
            sparse_matrix = embedder_instance.embed(self.dummy_texts)
            self.assertIsInstance(sparse_matrix, csr_matrix)

    # ----------- 🟣 MilvusBGEM3Embedder (Sparse) -----------
    def test_milvus_sparse_embedder_embed_returns_csr_array(self):
        """Test MilvusBGEM3Embedder embed method produces sparse matrix."""
        dummy_sparse = csr_matrix(np.eye(3))
        with patch.object(embedder, "BGEM3EmbeddingFunction"), \
             patch.object(MilvusBGEM3Embedder, "embed", return_value=dummy_sparse):
            embedder_instance = MilvusBGEM3Embedder()
            sparse_matrix = embedder_instance.embed(self.dummy_texts)
            self.assertIsInstance(sparse_matrix, csr_matrix)

    # ----------- 🟤 BaseEmbedder Factory -----------
    def test_base_embedder_from_default_dense(self):
        """Test BaseEmbedder.from_default returns DenseEmbedder."""
        with patch.object(embedder, "model_config", {"bert-base-uncased": {"type": "dense"}}), \
             patch.object(DenseEmbedder, "from_default", return_value="DenseInstance"):
            result = BaseEmbedder.from_default("bert-base-uncased")
            self.assertEqual(result, "DenseInstance")

    def test_base_embedder_from_default_sparse(self):
        """Test BaseEmbedder.from_default returns SparseEmbedder."""
        with patch.object(embedder, "model_config", {"BAAI/bge-m3": {"type": "sparse"}}), \
             patch.object(SparseEmbedder, "from_default", return_value="SparseInstance"):
            result = BaseEmbedder.from_default("BAAI/bge-m3")
            self.assertEqual(result, "SparseInstance")

    def test_base_embedder_from_default_invalid_type_raises(self):
        """Test BaseEmbedder.from_default raises error for unknown type."""
        with patch.object(embedder, "model_config", {"unknown-model": {"type": "invalid"}}):
            with self.assertRaises(ValueError):
                BaseEmbedder.from_default("unknown-model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
