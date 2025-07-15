from src.core.search_engine import Filter, SearchSpec
from typing import List
from src.core.schema import RouterConfig
from src.core.search_engine import SearchEngine

class BaseRouter: 
    def __init__(self):
        """
        Initializes the Router with a search specification.
        :param search_spec: An instance of SearchSpec that defines the search parameters.
        """
        self.specs: None

    def route(self, filter: Filter) -> int: 
        """
        Routes the filter to the appropriate search specification based on the filter criteria.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def load_specs(self, search_engines: List[SearchEngine]) -> None:
        """
        Loads the search specifications from the provided search engines.
        :param search_engines: A list of SearchEngine instances to load specifications from.
        """
        self.specs = [engine.spec() for engine in search_engines]
    
    @classmethod
    def from_default(cls, name, specs: List[SearchSpec]) -> 'BaseRouter':
        """
        Factory method to create a Router instance of specific type based on the name.
        """
        if name == "simple":
            return SimpleRouter(specs)
        elif name == "sparsity":
            return SparsityRouter(specs)
        else:
            raise ValueError(f"Unknown router type: {name}. Supported types: 'simple'.")
    
    @classmethod
    def from_config(cls, config: RouterConfig) -> 'BaseRouter':
        if config.type == "simple": 
            return SimpleRouter()
        
class SimpleRouter(BaseRouter): 
    def route(self, filter: Filter) -> int:
        return 0
        
class SparsityRouter(BaseRouter):
    def route(self, filter: Filter) -> int:
        """
        Routes the filter to the first search specification that matches the filter's sparsity criteria.
        """
        sparse_engines = [i for i, spec in enumerate(self.specs) if spec.optimal_for == 'strong'] #strong filters
        dense_engines = [i for i, spec in enumerate(self.specs) if spec.optimal_for == 'weak']
        for field in filter.must_fields(): 
            if getattr(filter, field) is not None: 
                return sparse_engines[0] if sparse_engines else dense_engines[0]
        return dense_engines[0] if dense_engines else sparse_engines[0]