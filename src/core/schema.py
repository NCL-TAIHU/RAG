from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field

# ---------------- Router and Reranker ----------------

class RouterConfig(BaseModel):
    type: Literal["simple"]

class RerankerConfig(BaseModel):
    type: Literal["identity", "auto_model"]

# ---------------- Embedder ----------------

class AutoModelEmbedderConfig(BaseModel):
    type: Literal["auto_model"]
    embedding_type: Literal["dense"]
    model_name: str

class BGEEmbedderConfig(BaseModel):
    type: Literal["bge"]
    embedding_type: Literal["sparse"]
    model_name: str

EmbedderConfig = Union[AutoModelEmbedderConfig, BGEEmbedderConfig]

# ---------------- Chunker ----------------
class LengthChunkerConfig(BaseModel):
    type: Literal["length_chunker"]
    chunk_size: int = 512
    overlap: int = 50

class SentenceChunkerConfig(BaseModel):
    type: Literal["sentence_chunker"]
    language: Literal["en", "zh"]

ChunkerConfig = Union[LengthChunkerConfig, SentenceChunkerConfig]

# ---------------- VectorSet ----------------
class VectorSetConfig(BaseModel):
    root: str
    dataset: str
    channel: str
    chunker: ChunkerConfig
    embedder: EmbedderConfig

# ---------------- Search Engine Configs ----------------

class MilvusConfig(BaseModel):
    type: Literal["milvus"]
    vector_set: VectorSetConfig    

class HybridMilvusConfig(BaseModel):
    type: Literal["hybrid_milvus"]
    sparse_vector_set: VectorSetConfig
    dense_vector_set: VectorSetConfig
    alpha: float = 0.5  # Weight for sparse vs dense scores

class ElasticSearchConfig(BaseModel):
    type: Literal["elasticsearch"]
    dataset: str
    es_host: str
    es_index: str

class SequentialConfig(BaseModel):
    type: Literal["sequential"]
    engines: List[Union[MilvusConfig, ElasticSearchConfig, HybridMilvusConfig]]

SearchEngineConfig = Union[
    MilvusConfig, 
    ElasticSearchConfig, 
    HybridMilvusConfig, 
    SequentialConfig
]


# ---------------- App Config ----------------

class AppConfig(BaseModel):
    '''
    A image of an existing app that is ready to be activated. 
    Front-end form data has to be enriched by backend with weave urls, ids, timestamps, and vector store roots to reach this stage. 
    '''
    id: str 
    name: str
    dataset: str
    description: Optional[str] = None

    search_engines: List[SearchEngineConfig]
    router: RouterConfig
    reranker: RerankerConfig
    max_files: int = 1000000  # For memory safety, default to 1 million

    weave_url: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None