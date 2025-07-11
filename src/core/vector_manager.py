from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from src.core.document import Document
from src.core.vector_store import DenseVS, SparseVS, BaseVS, VSMetadata
from src.core.embedder import DenseEmbedder, SparseEmbedder, BaseEmbedder
from scipy.sparse import csr_array
from src.core.util import get_first_content
from src.core.chunker import BaseChunker

class VectorManager:
    def __init__(self, vector_store: BaseVS, embedder: BaseEmbedder, chunker: BaseChunker, dataset: str, channel: str):
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker
        self.dataset = dataset
        self.channel = channel
        self.use_store = True

        assert embedder.name() == vector_store.meta().model, \
            f"Embedder model name ({embedder.name()}) does not match vector store model ({vector_store.meta().model})"
        assert dataset == vector_store.meta().dataset, \
            f"Dataset ({dataset}) does not match vector store dataset ({vector_store.meta().dataset})"
        assert channel == vector_store.meta().channel, \
            f"Channel ({channel}) does not match vector store channel ({vector_store.meta().channel})"
        assert chunker.metadata() == vector_store.meta().chunker_meta, \
            f"Chunker metadata ({chunker.metadata()}) does not match vector store chunker" 
        
    def get_raw_embedding(self, texts: List[str]) -> Union[List[List[float]], csr_array]:
        """
        Returns the raw embeddings for the given texts.
        """
        return self.embedder.embed(texts)
    
    def get_doc_embeddings(self, docs: List[Document]) -> Dict[str, Union[List[List[float]], csr_array]]:
        """
        Returns:
            - Dense: Dict[doc_id, List[List[float]]]
            - Sparse: Dict[doc_id, csr_array]
        """
        # Extract raw strings for the configured channel
        raw_inputs: List[str] = []
        doc_ids: List[str] = []

        for doc in docs:
            content = doc.content()
            assert self.channel in content, f"Channel '{self.channel}' not found in document {doc.key()}"
            field = content[self.channel]
            raw_texts = field.contents  # List[str]
            raw_inputs.append(" ".join(raw_texts))  # one string per doc
            doc_ids.append(doc.key())

        # Chunk into lists per doc
        chunked_texts: List[List[str]] = self.chunker.chunk(raw_inputs)

        embeddings: Dict[str, Any] = {}

        to_embed = []
        to_embed_ids = []
        to_retrieve_ids = []

        for doc_id, chunks in zip(doc_ids, chunked_texts):
            if self.use_store and self.vector_store.has(doc_id):
                to_retrieve_ids.append(doc_id)
            else:
                to_embed_ids.append(doc_id)
                to_embed.append(chunks)

        # Load from store
        if to_retrieve_ids:
            retrieved = self.vector_store.retrieve(to_retrieve_ids)
            embeddings.update(retrieved)

        # Compute new embeddings
        if to_embed:
            flat_chunks = [chunk for chunks in to_embed for chunk in chunks]
            embedded = self.embedder.embed(flat_chunks)

            if self.vector_store.meta().embedding_type == "sparse":
                assert isinstance(embedded, csr_array), "Sparse embedder must return csr_array"
            else:
                assert isinstance(embedded, list) and isinstance(embedded[0], list), "Dense embedder must return List[List[float]]"
            pointer = 0
            for doc_id, chunks in zip(to_embed_ids, to_embed):
                n = len(chunks)
                embeddings[doc_id] = embedded[pointer: pointer + n]
                pointer += n

        return embeddings

    def get_vs_metadata(self) -> VSMetadata:
        """
        Returns the metadata of the vector store.
        """
        return self.vector_store.meta()
    
    def get_channel(self) -> str:
        return self.channel