'''
Stores and retrieves id, embedding pairs. 
'''
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from scipy.sparse import csr_array
from pydantic import BaseModel
import os
import json
import numpy as np
from typing import List, Dict, Optional
from scipy.sparse import csr_array, vstack, save_npz, load_npz
import yaml
from datetime import datetime
from src.core.chunker import ChunkerMetaData

model_config = yaml.safe_load(open("config/model.yml", "r"))
class VSMetadata(BaseModel):
    """
    Metadata for the vector store, including the type of embeddings used.
    """
    embedding_type: str  # e.g., "dense" or "sparse"
    dataset: str #the dataset this vector store is associated with
    channel: str # content channel 
    chunker_meta: ChunkerMetaData
    model: str # Name of the model used for embeddings
    description: Optional[str] = None  # Optional description of the vector store
    version: Optional[str] = "1.0"  # Version of the vector store schema
    created_at: str# Timestamp of creation
    updated_at: str# Timestamp of last update


class BaseVS(ABC):
    @abstractmethod
    def insert(self, batch: Dict[str, Any]):
        """
        Inserts a batch of document embeddings into the vector store.

        :param batch: A dictionary mapping doc_id to embeddings:
                      - Dense: Dict[str, List[List[float]]]
                      - Sparse: Dict[str, csr_array] (shape = [num_chunks, dim])
        """
        pass

    @abstractmethod
    def has(self, id: str) -> bool:
        """
        Checks whether the vector store contains a specific document.

        :param id: Document ID
        :return: True if present, False otherwise
        """
        pass

    @abstractmethod
    def retrieve(self, ids: List[str]) -> Any:
        """
        Retrieves embeddings for the given document IDs.

        :param ids: List of document IDs
        :requires: ids exist in the vector store
        :return:
            - Dense: Dict[str, List[List[float]]]
            - Sparse: Dict[str, csr_array] (shape = [num_chunks, dim])
        """
        pass

    @abstractmethod
    def save(self):
        """
        Persists the vector store to disk (includes vectors, index, metadata).
        """
        pass

    @abstractmethod
    def meta(self) -> 'VSMetadata':
        """
        Returns the metadata of this vector store.

        :return: VSMetadata instance
        """
        pass

    @classmethod
    def create(cls, type: str, root: str, metadata: 'VSMetadata') -> 'BaseVS':
        """
        Factory method for creating a new vector store instance.

        :param type: 'dense' or 'sparse'
        :param root: Root directory for saving data
        :param metadata: Metadata object
        :return: Instance of DenseVS or SparseVS
        """
        if type == "dense":
            return FileBackedDenseVS(root=root, metadata=metadata)
        elif type == "sparse":
            return FileBackedSparseVS(root=root, metadata=metadata)
        else:
            raise ValueError(f"Unknown vector store type: {type}. Supported: 'dense', 'sparse'.")

    @classmethod
    def from_existing(cls, root: str) -> Optional['BaseVS']:
        """
        Loads an existing vector store from the given root directory.

        :param root: Directory containing metadata.json
        :return: FileBackedDenseVS or FileBackedSparseVS, or None on failure
        """
        meta_path = os.path.join(root, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"[ERROR] Metadata file not found at {meta_path}")
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = VSMetadata(**json.load(f))

        if metadata.embedding_type == "dense":
            return FileBackedDenseVS.from_existing(root)
        elif metadata.embedding_type == "sparse":
            return FileBackedSparseVS.from_existing(root)
        else:
            print(f"[ERROR] Unknown embedding type in metadata: {metadata.embedding_type}")
            return None


class DenseVS(BaseVS):
    @abstractmethod
    def insert(self, batch: Dict[str, List[List[float]]]):
        """
        Inserts dense embeddings for multiple documents.

        :param batch: Dict of doc_id → List of chunk vectors (List[List[float]])
        """
        pass

    @abstractmethod
    def retrieve(self, ids: List[str]) -> Dict[str, List[List[float]]]:
        """
        Retrieves dense chunk embeddings per document.

        :param ids: List of doc_ids
        :return: Dict of doc_id → List of chunk vectors
        """
        pass


class SparseVS(BaseVS):
    @abstractmethod
    def insert(self, batch: Dict[str, csr_array]):
        """
        Inserts sparse embeddings for multiple documents.

        :param batch: Dict of doc_id → csr_array (shape: [num_chunks, dim])
        """
        pass

    @abstractmethod
    def retrieve(self, ids: List[str]) -> Dict[str, csr_array]:
        """
        Retrieves sparse chunk embeddings per document.

        :param ids: List of doc_ids
        :return: Dict of doc_id → csr_array (shape: [num_chunks, dim])
        """
        pass

class FileBackedDenseVS(DenseVS):
    def __init__(self, root: str, metadata: VSMetadata):
        self._metadata = metadata
        self.root = root
        self.vector_path = os.path.join(self.root, "vectors.jsonl")
        self.meta_path = os.path.join(self.root, "metadata.json")
        self.vectors: Dict[str, List[List[float]]] = {}  # doc_id -> List[chunk vectors]

    def has(self, id: str) -> bool:
        return id in self.vectors

    def insert(self, batch: Dict[str, List[List[float]]]):
        for doc_id, chunk_vectors in batch.items():
            self.vectors[doc_id] = chunk_vectors

    def retrieve(self, ids: List[str]) -> Dict[str, List[List[float]]]:
        assert all(doc_id in self.vectors for doc_id in ids), "Attempting to retrieve non-existent document ids"
        return {doc_id: self.vectors[doc_id] for doc_id in ids}

    def save(self):
        self._metadata.updated_at = datetime.now().isoformat()
        os.makedirs(self.root, exist_ok=True)

        with open(self.vector_path, "w", encoding="utf-8") as f:
            for doc_id, vectors in self.vectors.items():
                f.write(json.dumps({doc_id: vectors}) + "\n")

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata.model_dump(), f, indent=2)

    def meta(self) -> VSMetadata:
        return self._metadata

    @classmethod
    def from_existing(cls, root: str) -> Optional['FileBackedDenseVS']:
        try:
            # Find and load metadata first
            base_meta_path = os.path.join(root, "metadata.json")
            if not os.path.exists(base_meta_path):
                raise FileNotFoundError(f"Missing metadata.json in {root}")

            with open(base_meta_path, "r", encoding="utf-8") as f:
                metadata = VSMetadata(**json.load(f))

            obj = cls(root=root, metadata=metadata)
            obj.vector_path = os.path.join(root, "vectors.jsonl")
            obj.meta_path = os.path.join(root, "metadata.json")

            if os.path.exists(obj.vector_path):
                with open(obj.vector_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line)
                        obj.vectors.update(entry)

            return obj
        except Exception as e:
            print(f"[ERROR] Failed to load FileBackedDenseVS: {e}")
            return None

class FileBackedSparseVS(SparseVS):
    def __init__(self, root: str, metadata: VSMetadata):
        self.root = root
        self._metadata = metadata
        self.matrix_path = os.path.join(root, f"vectors_matrix.npz")
        self.index_path = os.path.join(root, f"index.jsonl")
        self.meta_path = os.path.join(root, "metadata.json")
        self.rows: Dict[str, csr_array] = {}  # doc_id → (n_chunks x dim) csr_array

    def has(self, id: str) -> bool:
        return id in self.rows

    def insert(self, batch: Dict[str, csr_array]):
        for doc_id, mat in batch.items():
            self.rows[doc_id] = mat

    def retrieve(self, ids: List[str]) -> Dict[str, csr_array]:
        assert all(doc_id in self.rows for doc_id in ids), "Attempting to retrieve non-existent document ids"
        return {
            doc_id: self.rows[doc_id] for doc_id in ids 
        }

    def save(self):
        self._metadata.updated_at = datetime.now().isoformat()
        os.makedirs(self.root, exist_ok=True)

        doc_order = list(self.rows.keys())
        mat_list = [self.rows[doc_id] for doc_id in doc_order]
        full_matrix = vstack(mat_list)
        save_npz(self.matrix_path, full_matrix)

        with open(self.index_path, "w", encoding="utf-8") as f:
            for doc_id in doc_order:
                n_rows = self.rows[doc_id].shape[0]
                f.write(json.dumps({doc_id: n_rows}) + "\n")

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata.model_dump(), f, indent=2)

    def meta(self) -> VSMetadata:
        return self._metadata

    @classmethod
    def from_existing(cls, root: str) -> Optional['FileBackedSparseVS']:
        try:
            meta_path = os.path.join(root, "metadata.json")
            matrix_path = os.path.join(root, "vectors_matrix.npz")
            index_path = os.path.join(root, "index.jsonl")
            
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Missing metadata.json in {root}")

            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = VSMetadata(**json.load(f))

            obj = cls(root=root, metadata=metadata)
            if os.path.exists(matrix_path) and os.path.exists(index_path):
                full_matrix = load_npz(matrix_path)
                row_ptr = 0
                with open(index_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry: Dict = json.loads(line)
                        for doc_id, n_rows in entry.items():
                            obj.rows[doc_id] = full_matrix[row_ptr:row_ptr + n_rows]
                            row_ptr += n_rows

            return obj
        except Exception as e:
            print(f"[ERROR] Failed to load FileBackedSparseVS from {root}: {e}")
            return None