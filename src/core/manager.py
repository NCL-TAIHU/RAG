from src.core.search_engine import Filter
from src.core.document import Document
from typing import List
from src.core.library import Library
from src.core.search_engine import Filter, SearchEngine
from src.core.router import Router
from src.core.reranker import Reranker
import logging 

logger = logging.getLogger('taihu')

class Manager: 
    """
    Operator class for managing a library and multiple search engines.
    Search engines are routed based on the filter criteria, returning a list of ids. 
    The library is used to retrieve the actual documents based on these ids.
    """
    def __init__(self, 
                 library: Library, 
                 search_engines: List[SearchEngine], 
                 reranker: Reranker, 
                 router_name: str = "simple"
        ): 
        self.library: Library = library
        self.reranker: Reranker = reranker
        self.search_engines: List[SearchEngine] = search_engines
        self.router: Router = Router.from_default(router_name, [engine.spec() for engine in search_engines])

    def fetch(self, query: str, filter: Filter, limit: int) -> List[Document]:
        engine = self.search_engines[self.router.route(filter)]
        ids = engine.search(query, filter, limit)
        docs = self.library.retrieve(ids)
        return self.reranker.rerank(query, docs)
    
    def insert(self, docs: List[Document]) -> None:
        self.library.insert(docs)
        for engine in self.search_engines: engine.insert(docs)
    
    def setup(self) -> None:
        self.library.clear()
        for engine in self.search_engines: engine.setup()