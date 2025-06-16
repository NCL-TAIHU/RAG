from src.core.collection import CollectionBuilder, CollectionConfig, FieldConfig, IndexConfig, CollectionOperator
from src.core.llm import Agent
from src.core.embedder import DenseEmbedder, SparseEmbedder, AutoModelEmbedder, BGEM3Embedder, MilvusBGEM3Embedder
from src.core.data import DataLoader
from src.core.prompt import PromptBuilder
from src.core.document import Document, NCLDocument
from src.core.filter import Filter, NCLFilter
from src.core.search_engine import SearchEngine, Filter, HybridSearchEngine, MilvusSearchEngine, ElasticSearchEngine
from src.core.library import Library, InMemoryLibrary, FilesLibrary
from src.utils.logging import setup_logger
from scipy.sparse import csr_array
from typing import List
import sys
from src.core.manager import Manager
from tqdm import tqdm
CHATBOT = "meta-llama/Llama-3.1-8B-Instruct"
DENSE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_EMBEDDER = "BAAI/bge-m3"
DATASET = "ncl"  # Default dataset to use
DOC_CLS = NCLDocument  # Default document class
FILT_CLS = NCLFilter  # Default filter class

logger = setup_logger(
    name = 'search_app',
    log_file = 'logs/output.log', 
    console = True,
    file = False,
    level = "DEBUG"  # Set to DEBUG for detailed logs
)

class SearchApp:
    '''
    A search application that uses a combination of dense and sparse embeddings to retrieve relevant documents.
    The static methods are contextually static, meaning that their functionality does not depend on the instance state, 
    but is suited for this specific context of searching and embedding documents. For example, 
    the database schema is defined statically, as it does not change per instance, but if there's another app, 
    it may have a different schema or embedding strategy.
    '''
    def __init__(self, dataloader: DataLoader, manager: Manager, max_files: int = 1000):
        """Initialize the SearchApp."""
        self.data_loader: DataLoader = dataloader 
        self.manager: Manager = manager
        self.max_files: int = max_files

    def setup(self):
        self.manager.setup()
        count = 0
        for documents in tqdm(self.data_loader.load(), desc="Embedding batches"):
            self.manager.insert(documents)
            count += len(documents)
            if count >= self.max_files:
                logger.info(f"Inserted {count} documents, stopping further insertion.")
                break
        logger.info(f"Total documents inserted: {count}")
        self.llm: Agent = Agent.from_vllm(CHATBOT)
    
    def search(
            self, 
            query: str, 
            filter: Filter = None,
            limit: int = None, 
        ): 
        return self.manager.fetch(query=query, filter=filter, limit=limit)
    
    def rag(self, query: str, results: List[Document]) -> dict:
        prompt = PromptBuilder().add_user_message(query).add_documents(results).build_prompt()
        generation = self.llm.generate(prompt)
        return {
            "results": results,
            "prompt": prompt,
            "generation": generation
        }
    
def interact(app: SearchApp): 
    print("\n📚 Welcome to the Interactive Search App!")
    print("Type your query and press Enter to search.")
    print("Type `:rag` to toggle RAG mode, `:topk <num>` to change result count, or `:exit` to quit.")
    
    rag_enabled = False
    top_k = 5

    while True:
        user_input = input("\n>>> ").strip()
        if user_input.lower() in {":exit", "exit", "quit"}:
            print("👋 Exiting. Goodbye!")
            break
        elif user_input.lower() == ":rag":
            rag_enabled = not rag_enabled
            print(f"RAG mode {'enabled' if rag_enabled else 'disabled'}.")
            continue
        elif user_input.startswith(":topk"):
            try:
                top_k = int(user_input.split()[1])
                print(f"Result limit set to {top_k}.")
            except (IndexError, ValueError):
                print("Usage: :topk <int>")
            continue
        elif user_input.startswith(":"):
            print("❓ Unknown command.")
            continue

        # Run search
        f = Filter()
        results = app.search(query=user_input, filter = f, limit=top_k)
        print("\n🔍 Search Results:")
        if not results:
            print("No results found.")
            continue

        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.key()}")
            for field, data in doc.content().items():
                print(f"   {field}: {data.contents}...")

        if rag_enabled:
            response = app.rag(user_input, results)
            print("\n💬 LLM Response:")
            print(response["generation"])

def test(app: SearchApp): 
    filters = [
        NCLFilter().set_fields(year=[109]), 
        NCLFilter().set_fields(category=["碩士"]),
        NCLFilter().set_fields(school_chinese=["國立中山大學"]),
        NCLFilter().set_fields(dept_chinese=["資訊工程學系"]),
        NCLFilter().set_fields(authors_chinese=['許佩鈴']),
        NCLFilter().set_fields(advisors_chinese=['魏大華', '陳洋元'])
    ]
    queries = [
        "深度學習",
        "自然語言處理",
        "機器學習",
        "資料挖掘",
        "人工智慧", 
        "計算機視覺",
    ]

    for query, filter in zip(queries, filters):
        print(f"\n🔍 Searching for: {query}")
        results = app.search(query=query, filter=filter, limit=5)
        if not results:
            print("No results found.")
            continue
        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.key()}")
            for field, data in doc.content().items():
                print(f"   {field}: {data.contents}...")

        response = app.rag(query, results)
        print("\n💬 LLM Response:")
        print(response["generation"])

def main():
    print("🔎 Initializing SearchApp...")
    dataloader = DataLoader.from_default(DATASET)
    library: Library = InMemoryLibrary()
    sparse_embedder: SparseEmbedder = BGEM3Embedder(model_name=SPARSE_EMBEDDER)
    dense_embedder: DenseEmbedder = AutoModelEmbedder(model_name=DENSE_EMBEDDER)
    engine1 = HybridSearchEngine(
        relational_search_engine=ElasticSearchEngine("https://localhost:9200", document_cls=DOC_CLS, filter_cls=FILT_CLS, es_index= "documents"),
        vector_search_engine=MilvusSearchEngine(sparse_embedder, dense_embedder, document_cls=DOC_CLS, filter_cls=FILT_CLS)
    )
    engine2 = MilvusSearchEngine(sparse_embedder, dense_embedder, document_cls=DOC_CLS, filter_cls=FILT_CLS)
    manager = Manager(library, [engine1, engine2], router_name="sparsity")
    app = SearchApp(dataloader, manager)
    app.setup()
    test(app)
    interact(app)

if __name__ == "__main__":
    main()