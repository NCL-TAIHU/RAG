from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
import os
import json
from scipy.sparse import csr_array, vstack, save_npz, load_npz
from src.core.interface import StoredObj
from src.core.schema import VectorSetConfig
from src.core.chunker import BaseChunker
from src.core.embedder import BaseEmbedder
from src.core.document import Document
from src.core.data import DataLoader
from tqdm import tqdm
import logging
logger = logging.getLogger('taihu')

class BaseVectorSet(StoredObj):
    def __init__(self, config: VectorSetConfig):
        self._config = config
        self.root = config.root
        self.chunker = BaseChunker.from_config(config.chunker)
        self.embedder = BaseEmbedder.from_config(config.embedder)
        self.channel = config.channel

    def config(self) -> VectorSetConfig:
        return self._config

    def upsert(self, docs: List[Document]):
        contents = self.chunker.chunk([" ".join(doc.channels()[self.channel].contents) for doc in docs])
        assert len(contents) == len(docs), "Chunker should return same number of chunks as documents"
        ids = [doc.key() for doc in docs]
        embeddings = self.embedder.embed_chunks(ids, contents)
        self.update(embeddings)

    def setup(self):
        logger.info(f"Setting up vector set at {self.root}")
        dataloader = DataLoader.from_default(self._config.dataset)
        need_insert = [doc for doc in dataloader.stream() if not self.has(doc.key())]
        BATCH_SIZE = 64
        logger.info(f"Need to insert {len(need_insert)} documents into vector set")
        for i in tqdm(range(0, len(need_insert), BATCH_SIZE), desc="Inserting documents"):
            batch = need_insert[i:i + BATCH_SIZE]
            self.upsert(batch)
        self.save()

    def save(self):
        logger.info(f"Saving vector set to {self.root}")
        os.makedirs(self.root, exist_ok=True)
        self._save_vectors()
        with open(os.path.join(self.root, "id.txt"), "w", encoding="utf-8") as f:
            f.write(self._config.id)

    @classmethod
    def from_config(cls, config: VectorSetConfig) -> 'BaseVectorSet':
        try:
            id_path = os.path.join(config.root, "id.txt")
            if not os.path.exists(id_path):
                raise FileNotFoundError(f"Missing id.txt in {config.root}")
            with open(id_path, "r", encoding="utf-8") as f:
                existing_id = f.read().strip()
            if existing_id != config.id:
                raise ValueError(f"ID mismatch: expected {config.id}, found {existing_id} in {id_path}")

            if config.embedder.embedding_type == "dense":
                obj = FileBackedDenseVS(config)
            else: 
                obj = FileBackedSparseVS(config)
            obj._load_vectors()
            return obj
        except Exception:
            logger.exception(f"Failed to load {cls.__name__} from {config.root}, creating new vector set.")
            os.makedirs(config.root, exist_ok=True)
            if config.embedder.embedding_type == "dense":
                obj = FileBackedDenseVS(config)
            else: 
                obj = FileBackedSparseVS(config)
            return obj

    @abstractmethod
    def update(self, embeddings: Dict[str, Union[List[List[float]], csr_array]]):
        pass

    @abstractmethod
    def has(self, id: str) -> bool:
        pass

    @abstractmethod
    def retrieve(self, ids: List[str]) -> Union[Dict[str, List[List[float]]], Dict[str, csr_array]]:
        pass

    @abstractmethod
    def _save_vectors(self):
        pass

    @abstractmethod
    def _load_vectors(self):
        pass

class FileBackedDenseVS(BaseVectorSet):
    def __init__(self, config: VectorSetConfig):
        super().__init__(config)
        self.vector_path = os.path.join(self.root, "vectors.jsonl")
        self.vectors: Dict[str, List[List[float]]] = {}

    def has(self, id: str) -> bool:
        return id in self.vectors

    def update(self, embeddings: Dict[str, List[List[float]]]):
        self.vectors.update(embeddings)

    def retrieve(self, ids: List[str]) -> Dict[str, List[List[float]]]:
        assert all(doc_id in self.vectors for doc_id in ids)
        return {doc_id: self.vectors[doc_id] for doc_id in ids}

    def _save_vectors(self):
        with open(self.vector_path, "w", encoding="utf-8") as f:
            for doc_id, vectors in self.vectors.items():
                f.write(json.dumps({doc_id: vectors}) + "\n")

    def _load_vectors(self):
        if not os.path.exists(self.vector_path):
            raise FileNotFoundError(f"Missing vectors.jsonl in {self.root}")
        with open(self.vector_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                self.vectors.update(entry)

class FileBackedSparseVS(BaseVectorSet):
    def __init__(self, config: VectorSetConfig):
        super().__init__(config)
        self.matrix_path = os.path.join(self.root, "vectors_matrix.npz")
        self.index_path = os.path.join(self.root, "index.jsonl")
        self.rows: Dict[str, csr_array] = {}

    def has(self, id: str) -> bool:
        return id in self.rows

    def update(self, embeddings: Dict[str, csr_array]):
        self.rows.update(embeddings)

    def retrieve(self, ids: List[str]) -> Dict[str, csr_array]:
        assert all(doc_id in self.rows for doc_id in ids)
        return {doc_id: self.rows[doc_id] for doc_id in ids}

    def _save_vectors(self):
        doc_order = list(self.rows.keys())
        mat_list = [self.rows[doc_id] for doc_id in doc_order]
        full_matrix = vstack(mat_list)
        save_npz(self.matrix_path, full_matrix)

        with open(self.index_path, "w", encoding="utf-8") as f:
            for doc_id in doc_order:
                n_rows = self.rows[doc_id].shape[0]
                f.write(json.dumps({doc_id: n_rows}) + "\n")

    def _load_vectors(self):
        if not os.path.exists(self.matrix_path) or not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Missing matrix or index file in {self.root}")

        full_matrix = load_npz(self.matrix_path)
        row_ptr = 0
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                for doc_id, n_rows in entry.items():
                    self.rows[doc_id] = full_matrix[row_ptr : row_ptr + n_rows]
                    row_ptr += n_rows
