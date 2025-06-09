from typing import List, Optional, Dict
from src.core.entity import Document
from src.core.embedder import SparseEmbedder, DenseEmbedder
from src.core.entity import Document
from src.core.collection import FieldConfig, IndexConfig, CollectionConfig, CollectionOperator, CollectionBuilder
from typing import List
from pymilvus import (
    DataType,
    Collection,
)
from pymilvus.client.abstract import Hits, Hit
from elasticsearch import Elasticsearch
from pydantic import BaseModel
import yaml

class Filter(BaseModel):
    """
    Class to handle filtering of documents based on metadata.
    """
    ids: Optional[List[str]] = None  # filtered results ids need to be in this list
    keywords: List[str] = [] #filtered results need to contain all keywords

class SearchSpec(BaseModel):
    '''
    Description of the strengths and weaknesses of a search engine, used by router to determine which search engine to use.
    '''
    name: str
    

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
    
    def search(self, query: str, filter: Filter, limit: Optional[int]) -> List[str]:
        """
        Searches for documents based on a natural language query and optional metadata filters.
        :param query: The natural language query to search for.
        :param filter: Optional metadata filters to apply to the search.
        :param
        limit: The maximum number of documents to return.
        :return: A list of document IDs that match the search criteria.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def spec(self) -> SearchSpec: 
        raise NotImplementedError("This method should be overridden by subclasses.")

class HybridSearchEngine(SearchEngine):
    """
    A search engine that combines both relational and vector search capabilities.
    It uses a relational search engine for metadata filtering and a vector search engine for semantic search.
    """
    def __init__(self, relational_search_engine: SearchEngine, vector_search_engine: SearchEngine):
        self.relational_search_engine = relational_search_engine
        self.vector_search_engine = vector_search_engine

    def setup(self):
        self.relational_search_engine.setup()
        self.vector_search_engine.setup()

    def insert(self, docs: List[Document]):
        self.relational_search_engine.insert(docs)
        self.vector_search_engine.insert(docs)

    def search(self, query: str, filter: Filter, limit: int = 10) -> List[str]:
        filtered_ids = self.relational_search_engine.search(query, filter)
        subset_filter = Filter(ids=filtered_ids, keywords=filter.keywords)
        return self.vector_search_engine.search(query, subset_filter, limit=limit)
    
    def spec(self) -> SearchSpec:
        return SearchSpec(name="hybrid_search_engine")

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
    
    def spec(self) -> SearchSpec:
        return SearchSpec(name="milvus_search_engine")

class SQLiteSearchEngine(SearchEngine):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None

    def setup(self):
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
        cursor = self.connection.cursor()
        for doc in docs:
            cursor.execute('''
                INSERT OR REPLACE INTO documents (id, abstract, content, keywords)
                VALUES (?, ?, ?, ?)
            ''', (doc.id, doc.abstract, doc.content, ','.join(doc.keywords)))
        self.connection.commit()

    def search(self, query: str, filter: Filter, limit: int = 10) -> List[str]:
        cursor = self.connection.cursor()
        sql_query = '''
            SELECT id FROM documents
            WHERE 1=1'''
        params = []

        if filter.ids:
            sql_query += ' AND id IN ({})'.format(','.join(['?'] * len(filter.ids)))
            params.extend(filter.ids)

        if filter.keywords:
            for kw in filter.keywords:
                sql_query += ' AND keywords LIKE ?'
                params.append(f"%{kw}%")

        sql_query += ' LIMIT ?'
        params.append(limit)
        cursor.execute(sql_query, params)
        return [row[0] for row in cursor.fetchall()]
    
    def spec(self) -> SearchSpec:
        return SearchSpec(name="sqlite_search_engine")
    
class ElasticSearchEngine(SearchEngine):
    def __init__(self, es_host: str, es_index: str):
        config = yaml.safe_load(open("config/elastic_search.yml", "r"))
        self.es = Elasticsearch([es_host], 
                                basic_auth=("elastic", config["password"]),
                                verify_certs=True,
                                ca_certs=config["ca_certs"] 
                            ) 
        self.es_index = es_index

    def setup(self):
        # Always drop and recreate the index to ensure it's clean
        if self.es.indices.exists(index=self.es_index):
            self.es.indices.delete(index=self.es_index)
        self.es.indices.create(index=self.es_index)

    def insert(self, docs: List[Document]):
        for doc in docs:
            self.es.index(index=self.es_index, id=doc.id, body=doc.dict())

    def search(self, query: str, filter: Filter, limit: int = 10000) -> List[str]:
        must_filters = []
        keyword_filters = []

        for kw in filter.keywords:
            keyword_filters.append({"match_phrase": {"keywords": kw}})

        es_query = {
            "query": {
                "bool": {
                    "must": must_filters + keyword_filters,
                    "filter": []
                }
            },
            "size": limit
        }

        if filter.ids:
            es_query["query"]["bool"]["filter"].append({"terms": {"_id": filter.ids}})

        response = self.es.search(index=self.es_index, body=es_query)
        return [hit["_id"] for hit in response["hits"]["hits"]]
    
    def spec(self) -> SearchSpec:
        return SearchSpec(name="elastic_search_engine")
