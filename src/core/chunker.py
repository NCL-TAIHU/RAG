from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.core.document import Document
from pydantic import BaseModel

class ChunkerMetaData(BaseModel):
    """
    Metadata for chunking operations, including the chunker type and any additional parameters.
    """
    chunker_type: str  # e.g., "length", "semantic"
    params: Dict[str, Any] = {}  # Additional parameters for the chunker


class BaseChunker(ABC):
    """
    Abstract interface for extracting chunked semantic strings from documents.
    Each document can have multiple chunks (e.g., from a single channel or multiple channels).
    """

    @abstractmethod
    def chunk(self, docs: List[str]) -> List[List[str]]:
        """

        Example return:
            [
                ["doc1_chunk1", "doc1_chunk2"],
                ["doc2_chunk1"],
                ...
            ]
        """
        pass

    @abstractmethod
    def metadata(self) -> ChunkerMetaData:
        """
        Returns metadata about the chunking operation, such as the type of chunker used and any parameters.
        """
        pass

class LengthChunker(BaseChunker):
    """
    Simple chunker that splits documents into chunks of a specified length.
    """

    def __init__(self, chunk_length: int = 512):
        self.chunk_length = chunk_length

    def chunk(self, docs: List[str]) -> List[List[str]]:
        """
        Splits each document into chunks of the specified length.
        """
        return [[doc[i:i + self.chunk_length] for i in range(0, len(doc), self.chunk_length)] for doc in docs]
    
    def metadata(self) -> ChunkerMetaData:
        """
        Returns metadata about the chunking operation.
        """
        return ChunkerMetaData(chunker_type="length", params={"chunk_length": self.chunk_length})

class SentenceChunker(BaseChunker):
    """
    Chunker that splits documents into sentences.
    This is a placeholder for more complex sentence-based chunking logic.
    """

    def __init__(self, language: str = "en"):
        self.language = language
        assert language in ["en", "zh"], "Only English and Chinese are supported for sentence chunking"

    def chunk(self, docs: List[str]) -> List[List[str]]:
        """
        Splits each document into sentences.
        Placeholder implementation; actual implementation would use NLP libraries.
        """
        if self.language == "en":
            return [[sentence.strip() for sentence in doc.split('.')] for doc in docs]
        elif self.language == "zh": 
            return [[sentence.strip() for sentence in doc.split('。')] for doc in docs]

    def metadata(self) -> ChunkerMetaData:
        """
        Returns metadata about the chunking operation.
        """
        return ChunkerMetaData(chunker_type="sentence", params={"language": self.language})  
    