from src.core.schema import (
    VectorSetConfig, 
    LengthChunkerConfig, 
    AutoModelEmbedderConfig, 
    SentenceChunkerConfig, 
    EmbedderConfig, 
    AutoModelEmbedderConfig, 
    BGEEmbedderConfig, 
    ChunkerConfig, 
    RouterConfig, 
    RerankerConfig, 
    MilvusConfig, 
    HybridMilvusConfig, 
    ElasticSearchConfig,
    SequentialConfig, 
    SearchEngineConfig, 
    AppConfig
)
from typing import List
from src.core.document import Document
import os
import hashlib


def deterministic_get_id(key: str) -> str:
    """
    Generate a short deterministic ID (SHA1-based) from any string input.
    Returns the first 10 characters of the SHA1 hex digest.
    """
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


ROOT = f"_tests/storage/vector_set"

DATASETS = [
    "ncl", 
    "litsearch"
]

CHUNKERS: List[ChunkerConfig]= [
    LengthChunkerConfig(
        type="length_chunker",
        chunk_size=512,
        overlap=50
    ), 
    SentenceChunkerConfig(
        type="sentence_chunker",
        language = "en", 
    )
]
EMBEDDERS: List[EmbedderConfig] = [
    AutoModelEmbedderConfig(
        type="auto_model",
        embedding_type="dense",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ), 
    BGEEmbedderConfig(
        type="bge",
        embedding_type="sparse",
        model_name="BAAI/bge-m3"
    )
]

# NCL_DENSE_VS = VectorSetConfig(
#     id=ID,
#     root=f"_tests/storage/vector_set/{ID}", 
#     dataset="ncl", 
#     channel="abstract_chinese", 
#     chunker=LengthChunkerConfig(
#         type="length_chunker",
#         chunk_size=512,
#         overlap=50
#     ), 
#     embedder=AutoModelEmbedderConfig(
#         type="auto_model",
#         embedding_type="dense",
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
# )

VECTOR_SETS = [
    VectorSetConfig(
        id = id,
        root = os.path.join(ROOT, id),
        dataset = dataset,
        channel = channel, 
        chunker = chunker,
        embedder = embedder
    )
    for dataset in DATASETS
    for channel in Document.from_dataset(dataset).channels_schema().keys()
    for chunker in CHUNKERS
    for embedder in EMBEDDERS
    for id in [deterministic_get_id(f"{str(dataset)}_{str(channel)}_{str(chunker)}_{str(embedder)}")]
]

DENSE_VECTOR_SETS = [
    vs for vs in VECTOR_SETS if vs.embedder.embedding_type == "dense"
]

SPARSE_VECTOR_SETS = [
    vs for vs in VECTOR_SETS if vs.embedder.embedding_type == "sparse"
]

ROUTERS = [
    RouterConfig(type="simple"),
]

RERANKERS = [
    RerankerConfig(type="identity"),
]

VECTOR_ENGINES: List[SearchEngineConfig] = [
    MilvusConfig(
        type="milvus", 
        vector_set=vs
    ) for vs in DENSE_VECTOR_SETS + SPARSE_VECTOR_SETS
] + [
    HybridMilvusConfig(
        type="hybrid_milvus",
        sparse_vector_set=sparse_vs,
        dense_vector_set=dense_vs,
        alpha=0.5
    ) 
    for sparse_vs in SPARSE_VECTOR_SETS 
    for dense_vs in DENSE_VECTOR_SETS 
    if sparse_vs.channel == dense_vs.channel and sparse_vs.chunker == dense_vs.chunker
] 

STRUCTURED_ENGINES = [
    ElasticSearchConfig(
        type="elasticsearch",
        dataset=dataset,
        es_host="localhost",
        es_index=f"{dataset}_index"
    ) for dataset in DATASETS
]

ENGINES = VECTOR_ENGINES + STRUCTURED_ENGINES

ENGINES += [
    SequentialConfig(
        type="sequential",
        engines=[seng, veng] 
    )
    for seng in STRUCTURED_ENGINES
    for veng in VECTOR_ENGINES 
]

def is_valid_combo(
    dataset: str, 
    engine: SearchEngineConfig, 
) -> bool:
    """
    Validates if the combination of dataset, engine, router, and reranker is valid.
    """
    if isinstance(engine, MilvusConfig) and engine.vector_set.dataset != dataset:
        return False
    if isinstance(engine, HybridMilvusConfig):
        if engine.sparse_vector_set.dataset != dataset or engine.dense_vector_set.dataset != dataset:
            return False
    if isinstance(engine, ElasticSearchConfig) and engine.dataset != dataset:
        return False
    return True

APPS = [
    AppConfig(
        id=deterministic_get_id(f"{dataset}_{str(engine)}_{str(router)}_{str(reranker)}"),
        name=f"{dataset} App with {engine.type} Engine",
        dataset=dataset,
        description=f"App for {dataset} with {engine.type} engine, {router.type} router, and {reranker.type} reranker",
        search_engines=[engine],
        router=router,  # Assuming a single router for simplicity
        reranker=reranker  # Assuming a single reranker for simplicity
    )
    for dataset in DATASETS
    for engine in ENGINES
    for router in ROUTERS
    for reranker in RERANKERS
    if is_valid_combo(dataset, engine)
]