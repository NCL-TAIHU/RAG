from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.core.document import Document
from src.core.vector_store import DenseVS, SparseVS, BaseVS
from src.core.embedder import DenseEmbedder, SparseEmbedder, BaseEmbedder
from scipy.sparse import csr_array, vstack

class BaseVectorManager(ABC):
    def __init__(self, vector_store: BaseVS, embedder: BaseEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder
        assert embedder.name() == vector_store.meta().model, "Embedder model name does not match vector store model."

    @abstractmethod
    def get_doc_embeddings(self, docs: List[Document]) -> Any:
        pass

    def _get_first_content(self, doc: Document) -> str:
        """
        Helper method to extract the first content from a Document.
        If the document has no content, returns an empty string.
        """
        if doc.content():
            first_field = next(iter(doc.content().values()))
            if first_field.contents:
                return first_field.contents[0]
            else: 
                #first field exists but is empty
                return ""
        return ""

    def _collect_embeddings(self, docs: List[Document]) -> Dict[int, Any]:
        """
        Shared logic to retrieve or compute embeddings, returning a dict from original index → embedding.
        """
        embeddings_by_index = {}
        to_embed:List[Document] = []
        to_embed_indices = []
        to_retrieve_ids = []
        to_retrieve_indices = []

        for i, doc in enumerate(docs):
            doc_id = doc.key()
            if self.vector_store.has(doc_id):
                to_retrieve_ids.append(doc_id)
                to_retrieve_indices.append(i)
            else:
                to_embed.append(doc)
                to_embed_indices.append(i)

        if to_retrieve_ids:
            retrieved = self.vector_store.retrieve(to_retrieve_ids)
            for idx, emb in zip(to_retrieve_indices, retrieved):
                embeddings_by_index[idx] = emb

        if to_embed:
            texts = [self._get_first_content(doc) for doc in to_embed]
            computed = self.embedder.embed(texts)
            for idx, emb in zip(to_embed_indices, computed):
                embeddings_by_index[idx] = emb

        return embeddings_by_index

class DenseVectorManager(BaseVectorManager):
    def __init__(self, vector_store: DenseVS, embedder: DenseEmbedder):
        super().__init__(vector_store, embedder)
        
    def get_doc_embeddings(self, docs: List[Document]) -> List[List[float]]:
        embeddings_by_index = self._collect_embeddings(docs)
        return [embeddings_by_index[i] for i in range(len(docs))]

class SparseVectorManager(BaseVectorManager):
    def __init__(self, vector_store: SparseVS, embedder: SparseEmbedder):
        super().__init__(vector_store, embedder)

    def get_doc_embeddings(self, docs: List[Document]) -> csr_array:
        embeddings_by_index = self._collect_embeddings(docs)
        return vstack([embeddings_by_index[i] for i in range(len(docs))])