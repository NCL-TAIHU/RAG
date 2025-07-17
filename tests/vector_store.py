import unittest
import tempfile
import shutil
import os
import json
import numpy as np
from datetime import datetime
from scipy.sparse import csr_matrix
from src.core.vector_store import (
    FileBackedDenseVS, FileBackedSparseVS, VSMetadata, BaseVS
)
from src.core.chunker import ChunkerMetaData


def make_chunker_meta() -> ChunkerMetaData:
    """
    Create a dummy ChunkerMetaData for use in VSMetadata.
    """
    return ChunkerMetaData(
        chunker_type="sliding",
        chunk_size=512,
        overlap=64,
        strategy="by_sentence"
    )


def make_metadata(embedding_type: str) -> VSMetadata:
    """
    Create a VSMetadata instance with dummy data.
    """
    return VSMetadata(
        embedding_type=embedding_type,
        dataset="unit-test-dataset",
        channel="unit-test-channel",
        chunker_meta=make_chunker_meta(),
        model="dummy-model",
        description="Test vector store",
        version="1.0",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        """
        Create a temporary directory for test isolation.
        """
        self.temp_dir = tempfile.mkdtemp()
        print(f"\n[Setup] Created temp directory: {self.temp_dir}")

    def tearDown(self):
        """
        Clean up the temporary directory after tests.
        """
        shutil.rmtree(self.temp_dir)
        print(f"[Teardown] Removed temp directory: {self.temp_dir}")

    # ----------- 🔵 FileBackedDenseVS Tests -----------
    def test_dense_insert_and_retrieve(self):
        """Test inserting and retrieving dense vectors."""
        metadata = make_metadata("dense")
        vs = FileBackedDenseVS(root=self.temp_dir, metadata=metadata)
        dummy_vectors = {"doc1": [[0.1]*128, [0.2]*128]}
        vs.insert(dummy_vectors)
        self.assertTrue(vs.has("doc1"), "Inserted doc1 should exist in vector store.")
        retrieved = vs.retrieve(["doc1"])
        self.assertEqual(retrieved["doc1"], dummy_vectors["doc1"], "Retrieved vectors should match inserted data.")
        print("[Test] Dense insert and retrieve passed.")

    def test_dense_save_and_load(self):
        """Test saving dense vectors to disk and loading them back."""
        metadata = make_metadata("dense")
        vs = FileBackedDenseVS(root=self.temp_dir, metadata=metadata)
        vs.insert({"doc2": [[0.3]*128]})
        vs.save()
        print("[Test] Dense vectors saved.")

        loaded_vs = FileBackedDenseVS.from_existing(self.temp_dir)
        self.assertIsNotNone(loaded_vs, "Loaded vector store should not be None.")
        self.assertTrue(loaded_vs.has("doc2"), "Loaded vector store should contain doc2.")
        self.assertEqual(loaded_vs.retrieve(["doc2"])["doc2"], [[0.3]*128], "Loaded vectors should match saved data.")
        print("[Test] Dense save and load passed.")

    # ----------- 🟢 FileBackedSparseVS Tests -----------
    def test_sparse_insert_and_retrieve(self):
        """Test inserting and retrieving sparse matrices."""
        metadata = make_metadata("sparse")
        vs = FileBackedSparseVS(root=self.temp_dir, metadata=metadata)
        dummy_matrix = csr_matrix(np.eye(4))
        vs.insert({"docX": dummy_matrix})
        self.assertTrue(vs.has("docX"), "Inserted docX should exist in sparse vector store.")
        retrieved = vs.retrieve(["docX"])
        self.assertTrue((retrieved["docX"] != dummy_matrix).nnz == 0, "Retrieved sparse matrix should match inserted data.")
        print("[Test] Sparse insert and retrieve passed.")

    def test_sparse_save_and_load(self):
        """Test saving sparse matrices to disk and loading them back."""
        metadata = make_metadata("sparse")
        vs = FileBackedSparseVS(root=self.temp_dir, metadata=metadata)
        dummy_matrix = csr_matrix(np.eye(3))
        vs.insert({"docY": dummy_matrix})
        vs.save()
        print("[Test] Sparse vectors saved.")

        loaded_vs = FileBackedSparseVS.from_existing(self.temp_dir)
        self.assertIsNotNone(loaded_vs, "Loaded sparse vector store should not be None.")
        self.assertTrue(loaded_vs.has("docY"), "Loaded sparse vector store should contain docY.")
        loaded_matrix = loaded_vs.retrieve(["docY"])["docY"]
        self.assertTrue((loaded_matrix != dummy_matrix).nnz == 0, "Loaded sparse matrix should match saved data.")
        print("[Test] Sparse save and load passed.")

    # ----------- 🟣 BaseVS Factory Tests -----------
    def test_create_dense_vs_via_factory(self):
        """Test BaseVS factory creates FileBackedDenseVS."""
        metadata = make_metadata("dense")
        vs = BaseVS.create(type="dense", root=self.temp_dir, metadata=metadata)
        self.assertIsInstance(vs, FileBackedDenseVS, "Factory should return FileBackedDenseVS.")
        print("[Test] BaseVS.create for dense passed.")

    def test_create_sparse_vs_via_factory(self):
        """Test BaseVS factory creates FileBackedSparseVS."""
        metadata = make_metadata("sparse")
        vs = BaseVS.create(type="sparse", root=self.temp_dir, metadata=metadata)
        self.assertIsInstance(vs, FileBackedSparseVS, "Factory should return FileBackedSparseVS.")
        print("[Test] BaseVS.create for sparse passed.")

    def test_create_invalid_type_raises_value_error(self):
        """Test BaseVS factory raises ValueError for unknown type."""
        metadata = make_metadata("dense")
        with self.assertRaises(ValueError, msg="Creating with invalid type should raise ValueError."):
            BaseVS.create(type="invalid", root=self.temp_dir, metadata=metadata)
        print("[Test] BaseVS.create with invalid type raised ValueError as expected.")

    def test_from_existing_dense_vs(self):
        """Test BaseVS.from_existing loads FileBackedDenseVS."""
        metadata = make_metadata("dense")
        meta_path = os.path.join(self.temp_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata.model_dump(), f)
        vs = BaseVS.from_existing(self.temp_dir)
        self.assertIsInstance(vs, FileBackedDenseVS, "from_existing should load FileBackedDenseVS.")
        print("[Test] BaseVS.from_existing for dense passed.")

    def test_from_existing_sparse_vs(self):
        """Test BaseVS.from_existing loads FileBackedSparseVS."""
        metadata = make_metadata("sparse")
        meta_path = os.path.join(self.temp_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata.model_dump(), f)
        vs = BaseVS.from_existing(self.temp_dir)
        self.assertIsInstance(vs, FileBackedSparseVS, "from_existing should load FileBackedSparseVS.")
        print("[Test] BaseVS.from_existing for sparse passed.")

    def test_from_existing_missing_metadata_returns_none(self):
        """Test BaseVS.from_existing returns None if metadata.json missing."""
        vs = BaseVS.from_existing(self.temp_dir)
        self.assertIsNone(vs, "from_existing should return None if metadata.json is missing.")
        print("[Test] BaseVS.from_existing missing metadata passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
