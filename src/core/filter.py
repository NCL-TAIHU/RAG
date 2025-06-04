from typing import List
from elasticsearch import Elasticsearch

class Filter:
    '''
    Gives a list of ids representing the documents that match the filter criteria.
    '''
    def get(self) -> List[str]:
        """
        Returns a list of document IDs that match the filter criteria.
        This method should be implemented by subclasses to provide specific filtering logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
class DummyFilter(Filter):
    '''
    A dummy filter that returns all documents.
    '''
    def get(self) -> List[str]:
        """
        Returns an empty list, indicating no filtering is applied.
        """
        return ["1", "2", "3", "4", "5"]  # Example IDs, replace with actual logic if needed
    
class ElasticSearchFilter(Filter):
    '''
    A filter that uses Elasticsearch to filter documents based on a query.
    '''
    def __init__(self, es: Elasticsearch, index_name: str, query: dict):
        self.es = es
        self.index_name = index_name
        self.query = query

    def set_query(self, query: dict):
        self.query = query

    def get(self) -> List[str]:
        response = self.es.search(index=self.index_name, body=self.query, size=10000, _source=False)
        return [hit['_id'] for hit in response['hits']['hits']]
    
#TODO: Construct the elasticsearch database from documents and Construct the query from the criteria
#Milvus indexes the primary keys