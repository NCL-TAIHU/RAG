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

model_config = yaml.safe_load(open("config/model.yml", "r"))
class VSMetadata(BaseModel):
    """
    Metadata for the vector store, including the type of embeddings used.
    """
    embedding_type: str  # e.g., "dense" or "sparse"
    model: str # Name of the model used for embeddings
    description: Optional[str] = None  # Optional description of the vector store
    version: Optional[str] = "1.0"  # Version of the vector store schema
    created_at: Optional[str] = None  # Timestamp of creation
    updated_at: Optional[str] = None  # Timestamp of last update

    @classmethod
    def from_model(cls, model: str, description: Optional[str] = None) -> 'VSMetadata':
        embedding_type = model_config[model]["type"]
        #set to current time in ISO format
        now = datetime.now().isoformat()
        return cls(
            embedding_type=embedding_type,
            model=model,
            description=description,
            created_at=now,
            updated_at=now
        )

class BaseVS:
    def insert(self, ids: List[str], embeddings: Any):
        """
        Inserts a list of IDs and their corresponding embeddings into the vector store.
        :param ids: A list of document IDs to insert.
        :param embeddings: A list of embeddings corresponding to the provided IDs.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def has(self, id: str) -> bool: 
        """
        Checks if the vector store contains a specific ID.
        :param id: The document ID to check.
        :return: True if the ID exists in the vector store, False otherwise.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def save(self): 
        """
        Saves the current state of the vector store.
        This method should be overridden by subclasses if needed.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def retrieve(self, ids: List[str]) -> Any:
        """
        Retrieves embeddings from the vector store based on their IDs.
        :param ids: A list of document IDs to retrieve.
        :return: A list of embeddings corresponding to the provided IDs.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def meta(self) -> VSMetadata: 
        """
        Returns the metadata of the vector store.
        :return: An instance of VSMetadata containing information about the vector store.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @classmethod
    def from_default(cls, type: str, root: str, metadata: VSMetadata) -> 'BaseVS':
        if type == "dense":
            return FileBackedDenseVS(root=root, metadata=metadata)
        elif type == "sparse":
            return FileBackedSparseVS(root=root, metadata=metadata)
        else:
            raise ValueError(f"Unknown vector store type: {type}. Supported: 'dense', 'sparse'.")

class DenseVS(BaseVS):
    def insert(self, ids, embeddings: List[List[float]]): 
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def retrieve(self, ids: List[str]) -> List[List[float]]:
        """
        Retrieves a list of embeddings from the vector store based on their IDs.
        :param ids: A list of document IDs to retrieve.
        :return: A list of embeddings corresponding to the provided IDs.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class SparseVS(BaseVS):
    def insert(self, ids, embeddings: csr_array): 
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def retrieve(self, ids: List[str]) -> csr_array:
        """
        Retrieves a sparse matrix of embeddings from the vector store based on their IDs.
        :param ids: A list of document IDs to retrieve.
        :return: A sparse matrix corresponding to the provided IDs.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")



class FileBackedDenseVS(DenseVS):
    def __init__(self, root: str, metadata: VSMetadata):
        self.root = root
        self.vector_path = os.path.join(root, "vectors.jsonl")
        self.meta_path = os.path.join(root, "metadata.json")
        self.vectors: Dict[str, List[float]] = {}
        self._metadata = metadata

    def has(self, id: str) -> bool:
        return id in self.vectors

    def insert(self, ids: List[str], embeddings: List[List[float]]):
        for doc_id, emb in zip(ids, embeddings):
            self.vectors[doc_id] = emb

    def retrieve(self, ids: List[str]) -> List[List[float]]:
        return [self.vectors[doc_id] for doc_id in ids]

    def save(self):
        os.makedirs(self.root, exist_ok=True)
        with open(self.vector_path, "w", encoding="utf-8") as f:
            for doc_id, vector in self.vectors.items():
                f.write(json.dumps({"id": doc_id, "vector": vector}) + "\n")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata.model_dump(), f, indent=2)

    def meta(self) -> VSMetadata:
        return self._metadata

    @classmethod
    def from_existing(cls, root: str) -> Optional['FileBackedDenseVS']:
        meta_path = os.path.join(root, "metadata.json")
        vector_path = os.path.join(root, "vectors.jsonl")

        try: 
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Missing metadata.json in {root}")
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = VSMetadata(**json.load(f))

            obj = cls(root=root, metadata=metadata)

            if os.path.exists(vector_path):
                with open(vector_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line)
                        obj.vectors[entry["id"]] = entry["vector"]

            return obj
        except Exception as e:
            print(f"[ERROR] Failed to load FileBackedDenseVS from {root}: {e}")
            return None
    
class FileBackedSparseVS(SparseVS):
    def __init__(self, root: str, metadata: VSMetadata):
        self.root = root
        self.matrix_path = os.path.join(root, "vectors.npz")
        self.ids_path = os.path.join(root, "ids.txt")
        self.meta_path = os.path.join(root, "metadata.json")
        self.id_to_index: Dict[str, int] = {}
        self.matrix: Optional[csr_array] = None
        self._metadata = metadata

    def has(self, id: str) -> bool:
        return id in self.id_to_index

    def insert(self, ids: List[str], embeddings: csr_array):
        if self.matrix is None:
            self.matrix = embeddings
            self.id_to_index = {doc_id: i for i, doc_id in enumerate(ids)}
        else:
            self.matrix = vstack([self.matrix, embeddings])
            offset = len(self.id_to_index)
            for i, doc_id in enumerate(ids):
                self.id_to_index[doc_id] = offset + i

    def retrieve(self, ids: List[str]) -> csr_array:
        indices = [self.id_to_index[doc_id] for doc_id in ids]
        return self.matrix[indices]

    def save(self):
        os.makedirs(self.root, exist_ok=True)
        if self.matrix is not None:
            save_npz(self.matrix_path, self.matrix)
            with open(self.ids_path, "w", encoding="utf-8") as f:
                for doc_id in sorted(self.id_to_index, key=self.id_to_index.get):
                    f.write(doc_id + "\n")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata.model_dump(), f, indent=2)

    def meta(self) -> VSMetadata:
        return self._metadata

    @classmethod
    def from_existing(cls, root: str) -> Optional['FileBackedSparseVS']:
        meta_path = os.path.join(root, "metadata.json")
        matrix_path = os.path.join(root, "vectors.npz")
        ids_path = os.path.join(root, "ids.txt")

        try: 
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Missing metadata.json in {root}")
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = VSMetadata(**json.load(f))

            obj = cls(root=root, metadata=metadata)

            if os.path.exists(matrix_path):
                obj.matrix = load_npz(matrix_path)
            if os.path.exists(ids_path):
                with open(ids_path, "r", encoding="utf-8") as f:
                    obj.id_to_index = {line.strip(): i for i, line in enumerate(f)}
            return obj
        
        except Exception as e:
            print(f"[ERROR] Failed to load FileBackedSparseVS from {root}: {e}")
            return None
