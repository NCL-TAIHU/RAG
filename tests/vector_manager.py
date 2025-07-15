import unittest
from unittest.mock import MagicMock, patch
from src.core.vector_manager import VectorManager
from src.core.vector_store import VSMetadata
from src.core.chunker import ChunkerMetaData
from scipy.sparse import csr_matrix


class TestVectorManager(unittest.TestCase):
    def setUp(self):
        # 🌟 Patch model_config globally for all tests
        patcher = patch("src.core.vector_manager.model_config", {
            "bert-base-uncased": {"alias": "bert"}
        })
        self.addCleanup(patcher.stop)  # Clean up after each test
        self.mock_config = patcher.start()

        # ✅ Real ChunkerMetaData
        real_chunker_meta = ChunkerMetaData(
            chunker_type="length",
            params={"chunk_length": 512}
        )

        # ✅ VSMetadata with matching model alias
        self.metadata = VSMetadata(
            embedding_type="dense",
            dataset="test_dataset",
            channel="abstract",
            chunker_meta=real_chunker_meta,
            model="bert",  # 👈 must match alias in mocked config
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )

        # ✅ Mock BaseVS
        self.mock_vs = MagicMock()
        self.mock_vs.meta.return_value = self.metadata
        self.mock_vs.has.return_value = False
        self.mock_vs.retrieve.return_value = {"doc1": [[0.1, 0.2]]}

        # ✅ Mock BaseEmbedder
        self.mock_embedder = MagicMock()
        self.mock_embedder.name.return_value = "bert-base-uncased"
        self.mock_embedder.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        # ✅ Mock BaseChunker
        self.mock_chunker = MagicMock()
        self.mock_chunker.metadata.return_value = real_chunker_meta
        self.mock_chunker.chunk.return_value = [["chunk1", "chunk2"]]

        # ✅ Dummy Document
        self.mock_doc = MagicMock()
        self.mock_doc.content.return_value = {
            "abstract": MagicMock(contents=["Sentence one.", "Sentence two."])
        }
        self.mock_doc.key.return_value = "doc1"

    def test_init_metadata_consistency(self):
        """VectorManager raises assertion if metadata mismatches."""
        self.mock_vs.meta.return_value.model = "wrong_alias"
        with self.assertRaises(AssertionError):
            VectorManager(self.mock_vs, self.mock_embedder, self.mock_chunker, "test_dataset", "abstract")

    def test_get_raw_embedding_dense(self):
        """get_raw_embedding calls embedder and returns dense vectors."""
        vm = VectorManager(self.mock_vs, self.mock_embedder, self.mock_chunker, "test_dataset", "abstract")
        result = vm.get_raw_embedding(["Text one.", "Text two."])
        self.mock_embedder.embed.assert_called_once()
        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])

    def test_get_raw_embedding_sparse(self):
        """get_raw_embedding returns csr_matrix for sparse embedder."""
        self.metadata.embedding_type = "sparse"
        self.mock_embedder.embed.return_value = csr_matrix([[1, 0], [0, 1]])
        vm = VectorManager(self.mock_vs, self.mock_embedder, self.mock_chunker, "test_dataset", "abstract")
        result = vm.get_raw_embedding(["Doc1", "Doc2"])
        self.assertIsInstance(result, csr_matrix)

    def test_get_doc_embeddings_without_cache(self):
        """get_doc_embeddings embeds new docs and skips retrieve."""
        vm = VectorManager(self.mock_vs, self.mock_embedder, self.mock_chunker, "test_dataset", "abstract")
        result = vm.get_doc_embeddings([self.mock_doc])
        self.mock_chunker.chunk.assert_called_once()
        self.mock_embedder.embed.assert_called_once()
        self.assertIn("doc1", result)
        self.assertEqual(result["doc1"], [[0.1, 0.2], [0.3, 0.4]])

    def test_get_doc_embeddings_with_cache(self):
        """get_doc_embeddings retrieves from VS if cached."""
        self.mock_vs.has.return_value = True
        vm = VectorManager(self.mock_vs, self.mock_embedder, self.mock_chunker, "test_dataset", "abstract")
        result = vm.get_doc_embeddings([self.mock_doc])
        self.mock_vs.retrieve.assert_called_once()
        self.assertIn("doc1", result)

    def test_get_channel_returns_correct(self):
        """get_channel returns the configured channel."""
        vm = VectorManager(self.mock_vs, self.mock_embedder, self.mock_chunker, "test_dataset", "abstract")
        self.assertEqual(vm.get_channel(), "abstract")

    def test_get_vs_metadata_returns_correct(self):
        """get_vs_metadata returns VSMetadata."""
        vm = VectorManager(self.mock_vs, self.mock_embedder, self.mock_chunker, "test_dataset", "abstract")
        meta = vm.get_vs_metadata()
        self.assertEqual(meta, self.metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
