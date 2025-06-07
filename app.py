from src.core.collection import CollectionBuilder, CollectionConfig, FieldConfig, IndexConfig, CollectionOperator
from src.core.llm import LLMBuilder, generate
from src.core.embedder import DenseEmbedder, SparseEmbedder, AutoModelEmbedder, BGEM3Embedder, MilvusBGEM3Embedder
from src.core.data import DataLoader
from src.core.prompt import PromptBuilder
from src.core.entity import Document
from src.core.search_engine import SearchEngine, Filter
from src.core.library import Library, InMemoryLibrary, FilesLibrary
from src.utils.logging import setup_logger
from scipy.sparse import csr_array
from typing import List
import sys
from src.core.manager import BaseManager, MilvusElasticManager
from tqdm import tqdm
CHATBOT = "meta-llama/Llama-3.1-8B-Instruct"
DENSE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_EMBEDDER = "BAAI/bge-m3"
DATASET = "history"  # Default dataset to use

logger = setup_logger(
    name = 'search_app',
    log_file = 'logs/output.log', 
    console = True,
    file = False,
    level = "INFO"  # Set to DEBUG for detailed logs
)

class SearchApp:
    '''
    A search application that uses a combination of dense and sparse embeddings to retrieve relevant documents.
    The static methods are contextually static, meaning that their functionality does not depend on the instance state, 
    but is suited for this specific context of searching and embedding documents. For example, 
    the database schema is defined statically, as it does not change per instance, but if there's another app, 
    it may have a different schema or embedding strategy.
    '''
    def __init__(self, dataloader: DataLoader, manager: BaseManager):
        """Initialize the SearchApp."""
        self.data_loader: DataLoader = dataloader 
        self.manager: BaseManager = manager

    def setup(self):
        self.manager.setup()
        for documents in tqdm(self.data_loader.load(), desc="Embedding batches"):
            self.manager.insert(documents)
        self.llm = LLMBuilder.from_default(CHATBOT).build()    
    
    def search(
            self, 
            query: str, 
            filter: Filter = None,
            limit: int = None, 
        ): 
        return self.manager.fetch(query=query, filter=filter, limit=limit)
    
    def rag(self, query: str, results: List[Document]) -> dict:
        prompt = PromptBuilder().add_user_message(query).add_documents(results).build_prompt()
        generation = generate(self.llm, prompt)
        return {
            "results": results,
            "prompt": prompt,
            "generation": generation
        }
    
def main():
    print("🔎 Initializing SearchApp...")
    dataloader = DataLoader.from_default(DATASET)
    library: Library = InMemoryLibrary()
    sparse_embedder: SparseEmbedder = BGEM3Embedder(model_name=SPARSE_EMBEDDER)
    dense_embedder: DenseEmbedder = AutoModelEmbedder(model_name=DENSE_EMBEDDER)
    manager: BaseManager = MilvusElasticManager(library=library,
                                                sparse_embedder=sparse_embedder,
                                                dense_embedder=dense_embedder,
                                                elastic_host="https://localhost:9200",
                                                elastic_index="documents")
    app = SearchApp(dataloader, manager)
    app.setup()

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
        f = Filter(ids = None, keywords = [])
        results = app.search(query=user_input, filter = f, limit=top_k)
        print("\n🔍 Search Results:")
        if not results:
            print("No results found.")
            continue

        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.id}")
            print(f"     Abstract: {doc.abstract[:200]}...\n")

        if rag_enabled:
            response = app.rag(user_input, results)
            print("\n💬 LLM Response:")
            print(response["generation"])

if __name__ == "__main__":
    main()