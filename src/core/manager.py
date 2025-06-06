from src.core.search_engine import Filter
from src.core.entity import Document
from typing import List
from src.core.library import Library
from src.core.embedder import SparseEmbedder, DenseEmbedder
from src.core.search_engine import Filter, SearchEngine, MilvusSearchEngine, SQLiteSearchEngine

class BaseManager: 
    def __init__(self, library): 
        self.library: Library = library

    def fetch(self, query: str, filter: Filter, limit: int) -> List[Document]:
        ids = self.run_search(query, filter, limit)
        return self.library.retrieve(ids)
    
    def insert(self, docs: List[Document]) -> None:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def run_search(self, query: str, filter: Filter, limit: int) -> List[str]:
        """
        Runs a search query against the library.
        :param query: The search query string.
        :param filter: Optional metadata filters to apply to the search.
        :param limit: The maximum number of documents to return.
        :return: A list of Document objects that match the search criteria.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def setup(self) -> None:
        """
        Sets up the manager and its components.
        This method should be called before any other operations.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class HybridManager(BaseManager):
    """
    Manages the hybrid search engine, relational metadata filtering and vector search.
    """
    def __init__(self, 
                 library: Library, 
                 sparse_embedder: SparseEmbedder, 
                 dense_embedder: DenseEmbedder, 
                 sqlite_path: str = "sqlite.db"
        ):
        super().__init__(library)
        self.sparse_embedder = sparse_embedder
        self.dense_embedder = dense_embedder
        self.sqlite_path = sqlite_path

    def setup(self) -> None:
        """
        Sets up the hybrid manager and its components.
        This method should be called before any other operations.
        """
        # Initialize SQLite database or any other setup required for hybrid search
        self.sqlite_search_engine = SQLiteSearchEngine(self.sqlite_path)
        self.sqlite_search_engine.setup()
        self.milvus_search_engine = MilvusSearchEngine(self.sparse_embedder, self.dense_embedder)
        self.milvus_search_engine.setup()

    def insert(self, docs: List[Document]) -> None:
        # Insert documents into both SQLite and Milvus and Library
        self.sqlite_search_engine.insert(docs)
        self.milvus_search_engine.insert(docs)
        self.library.insert(docs)    
    
    def run_search(self, query: str, filter: Filter, limit: int) -> List[str]:
        filtered_ids = self.sqlite_search_engine.search(query, filter, limit)
        subset_filter = Filter(ids=filtered_ids, keywords=filter.keywords)
        return self.milvus_search_engine.search(query, subset_filter, limit)

class SQLiteManager(BaseManager):
    """
    Manages the SQLite search engine and relational metadata filtering.
    """
    def __init__(self, library: Library, sqlite_path: str = "sqlite.db"):
        super().__init__(library)
        self.sqlite_path = sqlite_path
        self.sqlite_search_engine = SQLiteSearchEngine(self.sqlite_path)

    def setup(self) -> None:
        """
        Sets up the SQLite manager and its components.
        This method should be called before any other operations.
        """
        self.sqlite_search_engine.setup()

    def insert(self, docs: List[Document]) -> None:
        # Insert documents into SQLite and Library
        self.sqlite_search_engine.insert(docs)
        self.library.insert(docs)

    def run_search(self, query: str, filter: Filter, limit: int) -> List[str]:
        return self.sqlite_search_engine.search(query, filter, limit)
    
class MilvusManager(BaseManager):
    """
    Manages the Milvus search engine and vector search.
    """
    def __init__(self, library: Library, 
                 sparse_embedder: SparseEmbedder, 
                 dense_embedder: DenseEmbedder
        ):
        super().__init__(library)
        self.sparse_embedder = sparse_embedder
        self.dense_embedder = dense_embedder
        self.milvus_search_engine = MilvusSearchEngine(self.sparse_embedder, self.dense_embedder)

    def setup(self) -> None:
        """
        Sets up the Milvus manager and its components.
        This method should be called before any other operations.
        """
        self.milvus_search_engine.setup()

    def insert(self, docs: List[Document]) -> None:
        # Insert documents into Milvus and Library
        self.milvus_search_engine.insert(docs)
        self.library.insert(docs)

    def run_search(self, query: str, filter: Filter, limit: int) -> List[str]:
        return self.milvus_search_engine.search(query, filter, limit)