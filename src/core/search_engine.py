from typing import List, Optional, Dict
from src.core.entity import Document
from src.core.filter import Filter
from src.core.embedder import SparseEmbedder, DenseEmbedder
from src.core.entity import Document
from src.core.collection import FieldConfig, IndexConfig, CollectionConfig, CollectionOperator, CollectionBuilder
from typing import List
from src.core.search_engine import SearchEngine
from pymilvus import (
    DataType,
    Collection,
)
from pymilvus.client.abstract import Hits, Hit
from elasticsearch import Elasticsearch
from pydantic import BaseModel

class Filter(BaseModel):
    """
    Class to handle filtering of documents based on metadata.
    """
    ids: List[str] = None  # List of document IDs to filter by
    keywords: List[str] = [] #filtered results need to contain all keywords

class SearchEngine: 
    '''
    Highest level class that inserts documents and retrieves answers based on natural language queries and metadata filters. 
    '''
    def setup(self) -> None:
        """
        Sets up the database connection and initializes necessary components.
        This method should be called before any other operations.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def insert(self, docs: List[Document]) -> None:
        """
        Inserts a list of documents into the database.
        :param docs: A list of Document objects to be inserted.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def search(self, query: str, filter: Filter, limit: int) -> List[str]:
        """
        Searches for documents based on a natural language query and optional metadata filters.
        :param query: The natural language query to search for.
        :param filter: Optional metadata filters to apply to the search.
        :param
        limit: The maximum number of documents to return.
        :return: A list of document IDs that match the search criteria.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class MilvusSearchEngine(SearchEngine): 
    def __init__(self, sparse_embedder: SparseEmbedder, dense_embedder: DenseEmbedder):
        """
        Initializes the MilvusDB with sparse and dense embedders.
        :param sparse_embedder: An instance of SparseEmbedder for sparse vector embeddings.
        :param dense_embedder: An instance of DenseEmbedder for dense vector embeddings.
        """
        self.sparse_embedder = sparse_embedder
        self.dense_embedder = dense_embedder
        self.config = CollectionConfig(
            collection_name="example_collection",
            fields=[
                FieldConfig(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldConfig(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldConfig(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=384), # 384 need to be parametrized based on the embedder
            ],
            indexes=[
                IndexConfig(
                field_name="sparse_vector",
                index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
                ),
                IndexConfig(
                    field_name="dense_vector",
                    index_params={
                        "index_type": "IVF_FLAT",     # predictable ANN behavior
                        "metric_type": "IP",
                        "params": {"nlist": 128}     # number of clusters
                    }
                )
            ]
        )

    def setup(self): 
        """
        Sets up the MilvusDB by building the collection.
        """
        builder = CollectionBuilder.from_config(self.config)
        builder.connect()
        self.collection = builder.build()
        self.operator = CollectionOperator(self.collection)

    def embed_documents(self, documents: List[Document]):
        abstracts = [doc.abstract for doc in documents]  # List of abstracts
        dense_embeddings = self.dense_embedder.embed(abstracts)
        sparse_embeddings = self.sparse_embedder.embed(abstracts)
        return dense_embeddings, sparse_embeddings
    
    def embed_query(self, query: str):
        dense_vector = self.dense_embedder.embed([query])[0]
        sparse_vector = self.sparse_embedder.embed([query])
        assert sparse_vector.shape[0] == 1, "Expected a single-row sparse vector"
        sparse_vector = sparse_vector._getrow(0)
        return dense_vector, sparse_vector
    
    def insert(self, documents: List[Document]):
        """
        Inserts a list of documents into the MilvusDB.
        :param documents: A list of Document objects to be inserted.
        """
        dense_embeddings, sparse_embeddings = self.embed_documents(documents)
        self.operator.buffered_insert([
            [doc.id for doc in documents],
            sparse_embeddings,
            dense_embeddings
        ])

    def search(self, query: str, filter: Filter, limit: int = 10) -> List[str]:
        '''
        ignore the filter for now, just return the top k results based on the query
        '''
        dense_vector, sparse_vector = self.embed_query(query)
        results = self.operator.search_hybrid(
            dense_vector, 
            sparse_vector, 
            alpha=0.5, 
            limit=limit, 
            subset_ids=filter.ids
        )
        hits: Hits = results[0]
        return [hit.fields.get("pk", "") for hit in hits]

class SQLiteSearchEngine(SearchEngine):
    #TODO: This is a dummy implementation, needs to be improved
    def __init__(self, db_path: str):
        """
        Initializes the SQLiteSearchEngine with the path to the SQLite database.
        :param db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.connection = None

    def setup(self):
        """
        Sets up the SQLite database connection and creates necessary tables.
        """
        import sqlite3
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                abstract TEXT,
                content TEXT,
                keywords TEXT
            )
        ''')
        self.connection.commit()

    def insert(self, docs: List[Document]):
        """
        Inserts a list of documents into the SQLite database.
        :param docs: A list of Document objects to be inserted.
        """
        cursor = self.connection.cursor()
        for doc in docs:
            cursor.execute('''
                INSERT OR REPLACE INTO documents (id, abstract, content, keywords)
                VALUES (?, ?, ?, ?)
            ''', (doc.id, doc.abstract, doc.content, ','.join(doc.keywords)))
        self.connection.commit()

    def search(self, query: str, filter: Filter, limit: int = 10) -> List[str]:
        """
        Searches for documents based on a natural language query and optional metadata filters.
        :param query: The natural language query to search for.
        :param filter: Optional metadata filters to apply to the search.
        :param limit: The maximum number of documents to return.
        :return: A list of document IDs that match the search criteria.
        """
        cursor = self.connection.cursor()
        
        sql_query = f'''
            SELECT id FROM documents
            WHERE abstract LIKE ? OR content LIKE ?
            LIMIT ?
        '''
        
        params = [f'%{query}%', f'%{query}%', limit]
        
        if filter.ids:
            sql_query += ' AND id IN ({})'.format(','.join(['?'] * len(filter.ids)))
            params.extend(filter.ids)

        cursor.execute(sql_query, params)
        
        results = cursor.fetchall()

        return [row[0] for row in results]

    
class ElasticSearchEngine(SearchEngine):
    #TODO: This is a dummy implementation, needs to be improved
    def __init__(self, es_host: str, es_index: str):
        """
        Initializes the ElasticSearchEngine with the host and index name.
        :param es_host: The host URL of the Elasticsearch instance.
        :param es_index: The name of the Elasticsearch index to use.
        """
        self.es = Elasticsearch([es_host])
        self.es_index = es_index

    def setup(self):
        """
        Sets up the Elasticsearch index if it does not exist.
        """
        if not self.es.indices.exists(index=self.es_index):
            self.es.indices.create(index=self.es_index)

    def insert(self, docs: List[Document]):
        """
        Inserts a list of documents into the Elasticsearch index.
        :param docs: A list of Document objects to be inserted.
        """
        for doc in docs:
            self.es.index(index=self.es_index, id=doc.id, body=doc.dict())

    def search(self, query: str, filter: Filter, limit: int = 10) -> List[str]:
        """
        Searches for documents based on a natural language query and optional metadata filters.
        :param query: The natural language query to search for.
        :param filter: Optional metadata filters to apply to the search.
        :param limit: The maximum number of documents to return.
        :return: A list of document IDs that match the search criteria.
        """
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"multi_match": {"query": query, "fields": ["abstract", "content"]}}
                    ],
                    "filter": []
                }
            },
            "size": limit
        }

        if filter.ids:
            body["query"]["bool"]["filter"].append({
                "terms": {"_id": filter.ids}
            })

        response = self.es.search(index=self.es_index, body=body)
        return [hit["_id"] for hit in response["hits"]["hits"]]