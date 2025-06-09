from src.core.search_engine import Filter, SearchSpec
from typing import List

class Router: 
    def __init__(self, specs: List[SearchSpec]):
        """
        Initializes the Router with a search specification.
        :param search_spec: An instance of SearchSpec that defines the search parameters.
        """
        self.specs: List[SearchSpec] = specs
        assert specs, "Router must be initialized with at least one SearchSpec."

    def route(self, filter: Filter) -> int: 
        """
        Routes the filter to the appropriate search specification based on the filter criteria.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @classmethod
    def from_default(cls, name, specs: List[SearchSpec]) -> 'Router':
        """
        Factory method to create a Router instance of specific type based on the name.
        """
        if name == "simple":
            return SimpleRouter(specs)
        else:
            raise ValueError(f"Unknown router type: {name}. Supported types: 'simple'.")
    
class SimpleRouter(Router): 
    def route(self, filter: Filter) -> int:
        return 0
        