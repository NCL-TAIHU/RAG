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
from src.core.schema import VectorSetConfig
import logging

model_config = yaml.safe_load(open("config/model.yml", "r"))
logger = logging.getLogger('taihu')

# class VSconfig(BaseModel):
#     """
#     config for the vector store, including the type of embeddings used.
#     """
#     embedding_type: str  # e.g., "dense" or "sparse"
#     dataset: str #the dataset this vector store is associated with
#     channel: str # content channel 
#     chunker_meta: Chunkerconfig
#     model: str # Name of the model used for embeddings
#     description: Optional[str] = None  # Optional description of the vector store
#     version: Optional[str] = "1.0"  # Version of the vector store schema
#     created_at: str# Timestamp of creation
#     updated_at: str# Timestamp of last update


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
        Persists the vector store to disk (includes vectors, index, config).
        """
        pass

    @abstractmethod
    def config(self) -> VectorSetConfig:
        """
        Returns the config of this vector store.

        :return: VectorSetConfig instance
        """
        pass

    @classmethod
    def from_config(cls, config: VectorSetConfig) -> 'BaseVS':
        """
        Factory method to create a vector store instance from a configuration.

        :param config: VectorSetConfig object containing parameters.
        :return: An instance of BaseVS.
        """
        embedding_type = config.embedder.embedding_type
        if embedding_type == "dense":
            return FileBackedDenseVS(
                root=config.root, 
                config=config
            )
        elif embedding_type == "sparse":
            return FileBackedSparseVS(
                root=config.root, 
                config=config
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}. Supported: 'dense', 'sparse'.")
    
    @classmethod
    def from_existing(cls, config: VectorSetConfig) -> Optional['BaseVS']:
        """
        Loads an existing vector store from the given root directory.

        :param root: Directory containing config.json
        :return: FileBackedDenseVS or FileBackedSparseVS, or None on failure
        """
        config_path = os.path.join(config.root, "config.json")
        if not os.path.exists(config_path):
            logger.exception(f" config file not found at {config_path}, returning none for vector store.")
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            config = VectorSetConfig(**json.load(f))

        if config.embedder.embedding_type == "dense":
            return FileBackedDenseVS.from_existing(config)
        elif config.embedder.embedding_type == "sparse":
            return FileBackedSparseVS.from_existing(config)
        else:
            logger.error(f"Unknown embedding type in config: {config.embedder.embedding_type}. Supported: 'dense', 'sparse'.")
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
    def __init__(self, root: str, config: VectorSetConfig):
        self._config = config
        self.root = root
        self.vector_path = os.path.join(self.root, "vectors.jsonl")
        self.config_path = os.path.join(self.root, "config.json")
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
        os.makedirs(self.root, exist_ok=True)

        with open(self.vector_path, "w", encoding="utf-8") as f:
            for doc_id, vectors in self.vectors.items():
                f.write(json.dumps({doc_id: vectors}) + "\n")

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._config.model_dump(), f, indent=2)

    def meta(self) -> VectorSetConfig:
        return self._config

    @classmethod
    def from_existing(cls, config: VectorSetConfig) -> Optional['FileBackedDenseVS']:
        try:
            # Find and load config first
            base_config_path = os.path.join(config.root, "config.json")
            if not os.path.exists(base_config_path):
                raise FileNotFoundError(f"Missing config.json in {config.root}")

            with open(base_config_path, "r", encoding="utf-8") as f:
                existin_config = VectorSetConfig(**json.load(f))

            assert config == existin_config, "Config mismatch when loading FileBackedDenseVS"

            obj = cls(root=config.root, config=config)
            obj.vector_path = os.path.join(config.root, "vectors.jsonl")
            obj.config_path = os.path.join(config.root, "config.json")

            if os.path.exists(obj.vector_path):
                with open(obj.vector_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line)
                        obj.vectors.update(entry)

            return obj
        except Exception as e:
            logging.exception(f"Failed to load FileBackedDenseVS: {e},returning None")
            return None

class FileBackedSparseVS(SparseVS):
    def __init__(self, root: str, config: VectorSetConfig):
        self.root = root
        self._config = config
        self.matrix_path = os.path.join(root, f"vectors_matrix.npz")
        self.index_path = os.path.join(root, f"index.jsonl")
        self.config_path = os.path.join(root, "config.json")
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
        os.makedirs(self.root, exist_ok=True)

        doc_order = list(self.rows.keys())
        mat_list = [self.rows[doc_id] for doc_id in doc_order]
        full_matrix = vstack(mat_list)
        save_npz(self.matrix_path, full_matrix)

        with open(self.index_path, "w", encoding="utf-8") as f:
            for doc_id in doc_order:
                n_rows = self.rows[doc_id].shape[0]
                f.write(json.dumps({doc_id: n_rows}) + "\n")

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._config.model_dump(), f, indent=2)

    def meta(self) -> VectorSetConfig:
        return self._config

    @classmethod
    def from_existing(cls, config: VectorSetConfig) -> Optional['FileBackedSparseVS']:
        try:
            config_path = os.path.join(config.root, "config.json")
            matrix_path = os.path.join(config.root, "vectors_matrix.npz")
            index_path = os.path.join(config.root, "index.jsonl")
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Missing config.json in {config.root}")

            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = VectorSetConfig(**json.load(f))

            assert config == existing_config, "Config mismatch when loading FileBackedSparseVS"
            obj = cls(root=config.root, config=config)
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
            logger.exception(f"Failed to load FileBackedSparseVS from {config.root}: {e}")
            return None