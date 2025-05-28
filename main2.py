from src.core.db import CollectionBuilder, CollectionConfig, FieldConfig, IndexConfig, CollectionManager
from src.core.llm import LLMConfig, get_llm
from src.core.embedder import DenseEmbedder, SparseEmbedder, AutoModelEmbedder, BGEM3Embedder, MilvusBGEM3Embedder
from src.core.data import DataLoader
from src.core.prompt import PromptBuilder
from src.core.entity import Document, MetaData
from src.utils.logging import setup_logger
from vllm import LLM, SamplingParams
from scipy.sparse import csr_array
from typing import List
from pymilvus.client.abstract import SearchResult
from pymilvus import (
    DataType,
    Collection,
)
ABSTRACT_PATH = "data/raw/abstracts"
CONTENT_PATH = "data/raw/contents"
KEYWORD_PATH = "data/raw/keywords"

CHATBOT = "meta-llama/Llama-3.1-8B-Instruct"
DENSE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_EMBEDDER = "BAAI/bge-m3"

logger = setup_logger(
    name = 'search_app',
    log_file = 'logs/output.log', 
    console = True,
    file = False,
)

def load_data():
    """Load data from the specified paths."""
    data_loader = DataLoader(ABSTRACT_PATH, CONTENT_PATH, KEYWORD_PATH, MAX_FILES=50)
    data_loader.load_data()
    documents = data_loader.get_documents()
    metadatas = data_loader.get_metadatas()
    assert len(documents) == len(metadatas), "Documents and metadata must have the same length"
    assert all(doc.id == meta.id for doc, meta in zip(documents, metadatas)), "Documents and metadata must match by ID"
    return documents, metadatas

def embed_abstracts(documents: List[Document], dense_embedder:DenseEmbedder, sparse_embedder:SparseEmbedder):
    abstracts = [doc.abstract for doc in documents]  # List of abstracts
    dense_embeddings = dense_embedder.embed(abstracts)
    sparse_embeddings = sparse_embedder.embed(abstracts)
    return dense_embeddings, sparse_embeddings
    
def construct_milvus_collection():
    db_config = CollectionConfig(
        collection_name="example_collection",
        fields=[
            FieldConfig(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldConfig(name="abstract", dtype=DataType.VARCHAR, max_length=10000),
            FieldConfig(name="keywords", dtype=DataType.VARCHAR, max_length=512),
            FieldConfig(name="content", dtype=DataType.VARCHAR, max_length=20000),
            FieldConfig(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldConfig(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        ],
        indexes=[
            IndexConfig(field_name="sparse_vector", index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}),
            IndexConfig(field_name="dense_vector", index_params={"index_type": "AUTOINDEX", "metric_type": "IP"})
        ]
    )
    collection_builder = CollectionBuilder.from_config(db_config)
    collection_builder.connect()
    collection = collection_builder.build()
    return collection

def insert_embeddings(
        collection: Collection, 
        documents: List[Document], 
        metadatas: List[MetaData], 
        dense_embeddings: List[List[float]], 
        sparse_embeddings: csr_array
    ):
    manager = CollectionManager(collection)
    assert len(documents) == len(metadatas) == len(dense_embeddings) == sparse_embeddings.shape[0], "All input lists must have the same length"
    manager.buffered_insert([
        [doc.id for doc in documents],
        [doc.abstract for doc in documents],
        [",".join(meta.keywords) for meta in metadatas], 
        [meta.content for meta in metadatas],
        sparse_embeddings,
        dense_embeddings
    ])

def retrieve(query, dense_embedder:DenseEmbedder, sparse_embedder: SparseEmbedder, collection):
    dense_vector = dense_embedder.embed([query])[0]
    sparse_vector = sparse_embedder.embed([query])._getrow(0)  # Get the first row as a csr_array
    manager = CollectionManager(collection)
    results = manager.search_hybrid(dense_vector, sparse_vector, alpha=0.5, limit=5)
    return results

def build_prompt(query: str, results: SearchResult) -> str:
    """Build a prompt for the LLM based on the query and retrieved results."""
    #convert results to a string format
    print(type(results[0]))
    hits = results[0]
    print(type(hits[0]))
    print(hits[0])
    results_str = "\n".join([f"{i+1}. {hit.fields.get('abstract', '')}" for i, hit in enumerate(hits)])
    prompt_builder = (PromptBuilder(system_prompt="Answer the question based on the retrieved documents.")
                      .add_user_message(query)
                      .add_retrieval_results(results_str))
    return prompt_builder.build_prompt()

def generate(prompt): 
    llm_config = LLMConfig(
        model= CHATBOT, 
        trust_remote_code=True,
        download_dir="/tmp/model_cache",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,  # 限制 GPU 記憶體使用率
        max_model_len=8192,          # 限制上下文長度
        quantization="fp8"
    )
    params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
    llm = get_llm(llm_config)
    response = llm.generate(prompt, sampling_params=params)
    return response[0].outputs[0].text if response and response[0].outputs else "No response generated."

if __name__ == "__main__":
    logger.info("loading data...")
    documents, metadatas = load_data()
    logger.info(f"data loaded successfully. Found {len(documents)} documents.")
    logger.info("initializing embedder...")
    dense_embedder = AutoModelEmbedder(model_name=DENSE_EMBEDDER)
    sparse_embedder = MilvusBGEM3Embedder()
    logger.info("embedder initialized successfully.")
    logger.info("embedding abstracts...")
    dense_embeddings, sparse_embeddings = embed_abstracts(documents, dense_embedder, sparse_embedder)
    print(f"Generated {len(dense_embeddings)} dense embeddings and {type(sparse_embeddings)} sparse embeddings.")
    print(f"sample dense embedding: {dense_embeddings[0]}")  
    print(f"sample sparse embedding: {sparse_embeddings[0]}") 
    print(f"sample sparse embedding type: {type(sparse_embeddings._getrow(0))}")  # should be csr_array
    print(f"sample sparse embedding chunk type: {type(sparse_embeddings[0:2])}")  # should be csr_array


    logger.info("abstracts embedded successfully.")
    logger.info("constructing Milvus collection...")
    collection = construct_milvus_collection()
    logger.info("Milvus collection constructed successfully.")

    logger.info("inserting embeddings into Milvus collection...")
    insert_embeddings(collection, documents, metadatas, dense_embeddings, sparse_embeddings)
    logger.info("Embeddings inserted successfully.")

    query = "日本將明治維新階段從西方引進的技術有哪些？"
    results = retrieve(query, dense_embedder = dense_embedder, sparse_embedder= sparse_embedder, collection=collection)
    print(f"Retrieved {len(results[0])} results for the query: {query}")
    # print(type(results))
    # print(results[0][0])
    prompt = build_prompt(query, results)
    print(f"Generated prompt: {prompt}")
    generation = generate(prompt)
    print(f"Generated response: {generation}")