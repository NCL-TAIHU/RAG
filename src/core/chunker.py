from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.core.document import Document
from pydantic import BaseModel
from src.core.schema import ChunkerConfig, LengthChunkerConfig, SentenceChunkerConfig

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

    @classmethod
    def from_config(cls, config: ChunkerConfig) -> 'BaseChunker':
        """
        Factory method to create a BaseChunker instance from a configuration.
        :param config: Configuration object containing chunker parameters.
        :return: An instance of BaseChunker.
        """
        if config.type == "length_chunker":
            return LengthChunker.from_config(config)
        elif config.type == "sentence_chunker":
            return SentenceChunker.from_config(config)
        else:
            raise ValueError(f"Unknown chunker type: {config.type}. Supported types: 'simple_chunker', 'sentence_chunker'.")

    @abstractmethod
    def config(self) -> ChunkerConfig:
        """
        Returns metadata about the chunking operation, such as the type of chunker used and any parameters.
        """
        pass

class LengthChunker(BaseChunker):
    """
    Simple chunker that splits documents into chunks of a specified length.
    """

    def __init__(self, chunk_length: int = 512, overlap: int = 50):
        self.chunk_length = chunk_length
        self.overlap = overlap
        assert chunk_length > overlap, "Chunk length must be greater than overlap"

    def chunk(self, docs: List[str]) -> List[List[str]]:
        """
        Splits each document into chunks of the specified length.
        """
        #return [[doc[i:i + self.chunk_length] for i in range(0, len(doc), self.chunk_length)] for doc in docs]
        return [[doc[i:i + self.chunk_length] for i in range(0, len(doc), self.chunk_length - self.overlap)] for doc in docs]
    
    @classmethod
    def from_config(cls, config: LengthChunkerConfig) -> 'LengthChunker':
        """
        Factory method to create a LengthChunker instance from a configuration.
        :param config: Configuration object containing chunker parameters.
        :return: An instance of LengthChunker.
        """
        return cls(chunk_length=config.chunk_size, overlap=config.overlap)

    def config(self) -> LengthChunkerConfig:
        """
        Returns metadata about the chunking operation, such as the type of chunker used and any parameters.
        """
        return LengthChunkerConfig(type="length_chunker", chunk_size=self.chunk_length, overlap=0)
    
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

    @classmethod
    def from_config(cls, config: SentenceChunkerConfig) -> 'SentenceChunker':
        """
        Factory method to create a SentenceChunker instance from a configuration.
        :param config: Configuration object containing chunker parameters.
        :return: An instance of SentenceChunker.
        """
        return cls(language=config.language)
    
    def config(self) -> SentenceChunkerConfig:
        """
        Returns metadata about the chunking operation, such as the type of chunker used and any parameters.
        """
        return SentenceChunkerConfig(type="sentence_chunker", language=self.language)