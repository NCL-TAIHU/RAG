from src.core.search_engine import Filter
from src.core.entity import Document
from typing import List
from src.core.library import Library
from src.core.embedder import SparseEmbedder, DenseEmbedder
from src.core.search_engine import Filter, SearchEngine, MilvusSearchEngine, SQLiteSearchEngine, ElasticSearchEngine
import logging 
logger = logging.getLogger(__name__)

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
    """Manages both relational and vector search engines for metadata filtering and vector search."""
    def __init__(
            self, 
            library: Library, 
            relational_search_engine: SearchEngine, 
            vector_search_engine: SearchEngine
        ):
        super().__init__(library)
        self.relational_search_engine = relational_search_engine
        self.vector_search_engine = vector_search_engine

    def setup(self) -> None:
        self.relational_search_engine.setup()
        self.vector_search_engine.setup()
    
    def insert(self, docs: List[Document]) -> None:
        self.relational_search_engine.insert(docs)
        self.vector_search_engine.insert(docs)
        self.library.insert(docs)

    def run_search(self, query: str, filter: Filter, limit: int) -> List[str]:
        logger.debug(f"doing relational search with filter: {filter}")
        filtered_ids = self.relational_search_engine.search(query, filter)
        subset_filter = Filter(ids=filtered_ids, keywords=filter.keywords)
        logger.debug(f"doing vector search with subset filter: {subset_filter}")
        return self.vector_search_engine.search(query, subset_filter, limit)

class MonolithManager(BaseManager):
    """
    Manages a single search engine for both vector search and metadata filtering.
    This class is suitable for simpler use cases where a single search engine suffices.
    """
    def __init__(self, library: Library, search_engine: SearchEngine):
        super().__init__(library)
        self.search_engine = search_engine

    def setup(self) -> None:
        self.search_engine.setup()

    def insert(self, docs: List[Document]) -> None:
        self.search_engine.insert(docs)
        self.library.insert(docs)

    def run_search(self, query: str, filter: Filter, limit: int) -> List[str]:
        return self.search_engine.search(query, filter, limit)
    
class MilvusElasticManager(HybridManager):
    """
    Manages the Milvus search engine and Elasticsearch for vector search and metadata filtering.
    """
    def __init__(self, 
                 library: Library, 
                 sparse_embedder: SparseEmbedder, 
                 dense_embedder: DenseEmbedder, 
                 elastic_host: str = "https://localhost:9200", 
                 elastic_index: str = "documents"
        ):
        super().__init__(
            library, 
            relational_search_engine=ElasticSearchEngine(es_host=elastic_host, es_index=elastic_index), 
            vector_search_engine=MilvusSearchEngine(sparse_embedder, dense_embedder)
        )
        
class MilvusSQLiteManager(HybridManager):
    def __init__(self, 
                 library: Library, 
                 sparse_embedder: SparseEmbedder, 
                 dense_embedder: DenseEmbedder, 
                 sqlite_path: str = "sqlite.db"
        ):
        super().__init__(
            library, 
            relational_search_engine=SQLiteManager(library, sqlite_path), 
            vector_search_engine=MilvusSearchEngine(sparse_embedder, dense_embedder)
        )
        

class SQLiteManager(MonolithManager):
    def __init__(self, library: Library, sqlite_path: str = "sqlite.db"):
        super().__init__(library, SQLiteSearchEngine(db_path=sqlite_path))
    
class MilvusManager(MonolithManager):
    """
    Manages the Milvus search engine and vector search.
    """
    def __init__(self, library: Library, 
                 sparse_embedder: SparseEmbedder, 
                 dense_embedder: DenseEmbedder
        ):
        super().__init__(
            library, 
            MilvusSearchEngine(sparse_embedder, dense_embedder)
        )