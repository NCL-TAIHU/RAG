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
import sys
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

class SearchApp:
    '''
    A search application that uses a combination of dense and sparse embeddings to retrieve relevant documents.
    The static methods are contextually static, meaning that their functionality does not depend on the instance state, 
    but is suited for this specific context of searching and embedding documents. For example, 
    the database schema is defined statically, as it does not change per instance, but if there's another app, 
    it may have a different schema or embedding strategy.
    '''
    def __init__(self, abstract_path, content_path, keyword_path, dense_embedder, sparse_embedder, max_files=50):
        """Initialize the SearchApp."""
        self.documents = []
        self.metadatas = []
        self.dense_embeddings = None
        self.sparse_embeddings = None
        self.dense_embedder = AutoModelEmbedder(model_name=dense_embedder)
        self.sparse_embedder = MilvusBGEM3Embedder()
        self.collection = None
        self.abstract_path = abstract_path
        self.content_path = content_path
        self.keyword_path = keyword_path
        self.max_files = max_files

    @staticmethod
    def load_data(abstract_path, content_path, keyword_path):
        """Load data from the specified paths."""
        data_loader = DataLoader(abstract_path, content_path, keyword_path, MAX_FILES=50)
        data_loader.load_data()
        documents = data_loader.get_documents()
        metadatas = data_loader.get_metadatas()
        assert len(documents) == len(metadatas), "Documents and metadata must have the same length"
        assert all(doc.id == meta.id for doc, meta in zip(documents, metadatas)), "Documents and metadata must match by ID"
        return documents, metadatas
    
    @staticmethod
    def embed_abstracts(documents, dense_embedder: DenseEmbedder, sparse_embedder: SparseEmbedder):
        abstracts = [doc.abstract for doc in documents]  # List of abstracts
        dense_embeddings = dense_embedder.embed(abstracts)
        sparse_embeddings = sparse_embedder.embed(abstracts)
        return dense_embeddings, sparse_embeddings
    
    @staticmethod
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
        return collection_builder.build()
    
    @staticmethod
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

    @staticmethod
    def build_prompt(query: str, results: SearchResult) -> str:
        """Build a prompt for the LLM based on the query and retrieved results."""
        #convert results to a string format
        hits = results[0]
        results_str = "\n".join([f"{i+1}. {hit.fields.get('abstract', '')}" for i, hit in enumerate(hits)])
        prompt_builder = (PromptBuilder(system_prompt="Answer the question based on the retrieved documents.")
                        .add_user_message(query)
                        .add_retrieval_results(results_str))
        return prompt_builder.build_prompt()

    @staticmethod
    def generate(llm: LLM, prompt: str) -> str: 
        params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
        response = llm.generate(prompt, sampling_params=params)
        return response[0].outputs[0].text if response and response[0].outputs else "No response generated."

    @staticmethod
    def construct_llm():
        llm_config = LLMConfig(
            model= CHATBOT, 
            trust_remote_code=True,
            download_dir="/tmp/model_cache",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,  # 限制 GPU 記憶體使用率
            max_model_len=8192,          # 限制上下文長度
            quantization="fp8"
        )
        return get_llm(llm_config)
    
    def setup(self):
        logger.info("Setting up SearchApp...")
        logger.info("loading data...")
        self.documents, self.metadatas = self.load_data(self.abstract_path, self.content_path, self.keyword_path)
        logger.info(f"data loaded successfully. Found {len(self.documents)} documents.")
        logger.info("embedding abstracts...")
        self.dense_embeddings, self.sparse_embeddings = self.embed_abstracts(self.documents, self.dense_embedder, self.sparse_embedder)
        logger.info("abstracts embedded successfully.")
        logger.info("constructing Milvus collection...")
        self.collection = self.construct_milvus_collection()
        logger.info("Milvus collection constructed successfully.")
        logger.info("inserting embeddings into Milvus collection...")
        self.insert_embeddings(self.collection, self.documents, self.metadatas, self.dense_embeddings, self.sparse_embeddings)
        logger.info("Embeddings inserted successfully.")
        logger.info("constructing LLM...")
        self.llm = self.construct_llm()
        logger.info("LLM constructed successfully.")
    
    def search(
            self, 
            query: str, 
            method: str = "hybrid_search" , 
            sparse_weight: float = None, 
            dense_weight: float = None,
            limit: int = None
        ): 
        assert method in ["dense_search", "sparse_search", "hybrid_search"], "Method must be one of: dense_search, sparse_search, hybrid_search"
        dense_vector = self.dense_embedder.embed([query])[0]
        sparse_vector = self.sparse_embedder.embed([query])
        assert sparse_vector.shape[0] == 1, "Expected a single-row sparse vector"
        sparse_vector = sparse_vector._getrow(0)
        
        manager = CollectionManager(self.collection)
        if method == "dense_search":
            results = manager.search_dense(dense_vector, limit=limit)
        elif method == "sparse_search":
            results = manager.search_sparse(sparse_vector, limit=limit)
        else:
            alpha = sparse_weight  / (dense_weight + sparse_weight) if dense_weight and sparse_weight else 0.5
            results = manager.search_hybrid(dense_vector, sparse_vector, alpha=alpha, limit=limit)
        prompt = self.build_prompt(query, results)
        generation = self.generate(self.llm, prompt)
        return {
            "results": results,
            "prompt": prompt,
            "generation": generation
        }
    
if __name__ == "__main__":
    search_app = SearchApp(
        abstract_path=ABSTRACT_PATH, 
        content_path=CONTENT_PATH, 
        keyword_path=KEYWORD_PATH, 
        dense_embedder=DENSE_EMBEDDER, 
        sparse_embedder=SPARSE_EMBEDDER
    )
    search_app.setup()
    logger.info("Search application initialized successfully")
    try:
        while True:
            query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
            if query.lower() in {"exit", "quit"}:
                logger.info("Exiting search application.")
                break

            method = input("Search method [dense_search / sparse_search / hybrid_search] (default=hybrid_search): ").strip() or "hybrid_search"
            if method not in {"dense_search", "sparse_search", "hybrid_search"}:
                print("Invalid method. Defaulting to hybrid_search.")
                method = "hybrid_search"

            sparse_weight = dense_weight = None
            if method == "hybrid_search":
                try:
                    sparse_weight = float(input("Enter sparse weight (default=1.0): ") or "1.0")
                    dense_weight = float(input("Enter dense weight (default=1.0): ") or "1.0")
                except ValueError:
                    print("Invalid weights. Using default of 1.0 for both.")
                    sparse_weight = dense_weight = 1.0

            try:
                limit = int(input("Number of results to retrieve (default=5): ") or "5")
            except ValueError:
                print("Invalid limit. Using default of 5.")
                limit = 5

            output = search_app.search(
                query=query,
                method=method,
                sparse_weight=sparse_weight,
                dense_weight=dense_weight,
                limit=limit
            )

            print("\n--- Search Results ---")
            for i, result in enumerate(output["results"]):
                print(f"[{i+1}] {result}")

            print("\n--- Prompt ---")
            print(output["prompt"])

            print("\n--- Generation ---")
            print(output["generation"])

    except KeyboardInterrupt:
        logger.info("\nSearch interrupted by user. Exiting.")
        sys.exit(0)