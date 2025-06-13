from typing import List, Optional, Dict, Type
from src.core.entity import Document, FieldType
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
    #filter clauses, filtered results need to match at least one of these clauses
    ids: Optional[List[str]] = None  
    years: Optional[List[int]] = None  
    categories: Optional[List[str]] = None  
    schools: Optional[List[str]] = None  
    depts: Optional[List[str]] = None  
    
    #must clauses
    keywords: List[str] = [] #filtered results need to contain all keywords
    authors: List[str] = []  # filtered results need to contain these authors
    advisors: List[str] = []  # filtered results need to contain these advisors

    def must_fields(self): 
        return ['keywords', 'authors', 'advisors']
    
    def filter_fields(self):
        return ['ids', 'years', 'categories', 'schools', 'depts']

class SearchSpec(BaseModel):
    '''
    Description of the strengths and weaknesses of a search engine, used by router to determine which search engine to use.
    '''
    name: str
    optimal_for: Optional[str] = None # e.g., "strong", "weak" filters 
    

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
        :param limit: The maximum number of documents to return.
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
        subset_filter = filter.model_copy(update={"ids": filtered_ids})
        return self.vector_search_engine.search(query, subset_filter, limit=limit)
    
    def spec(self) -> SearchSpec:
        return SearchSpec(name="hybrid_search_engine", optimal_for="strong")


class MilvusSearchEngine(SearchEngine):
    document_cls: Type[Document]
    filter_cls: Type[Filter]

    def __init__(
        self,
        sparse_embedder: SparseEmbedder,
        dense_embedder: DenseEmbedder,
        document_cls: Type[Document],
        filter_cls: Type[Filter],
        collection_name: str = "milvus_collection",
    ):
        self.sparse_embedder = sparse_embedder
        self.dense_embedder = dense_embedder
        self.document_cls = document_cls
        self.filter_cls = filter_cls

        # Build fields from document metadata schema
        fields = [
            FieldConfig(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100)
        ]
        for f in self.document_cls.metadata_schema():
            fields.append(FieldConfig(
                name=f.name,
                dtype=f.type.to_milvus_type(),
                max_length=f.max_len if f.type == FieldType.STRING else None
            ))
        fields += [
            FieldConfig(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldConfig(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_embedder.get_dim())
        ]

        self.config = CollectionConfig(
            collection_name=collection_name,
            fields=fields,
            indexes=[
                IndexConfig(
                    field_name="sparse_vector",
                    index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
                ),
                IndexConfig(
                    field_name="dense_vector",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": "IP",
                        "params": {"nlist": 128}
                    }
                )
            ]
        )

    def setup(self):
        builder = CollectionBuilder.from_config(self.config)
        builder.connect()
        self.collection = builder.build()
        self.operator = CollectionOperator(self.collection)

    def embed_query(self, query: str):
        dense = self.dense_embedder.embed([query])[0]
        sparse = self.sparse_embedder.embed([query])
        assert sparse.shape[0] == 1, "Expected a single-row sparse vector"
        return dense, sparse._getrow(0)

    def insert(self, documents: List[Document]):
        dense_texts = [f.contents[0] for doc in documents for f in doc.content() if f.name.startswith("abstract") and f.contents]
        dense_embeddings = self.dense_embedder.embed(dense_texts)
        sparse_embeddings = self.sparse_embedder.embed(dense_texts)

        insert_dict = {
            "pk": [doc.key() for doc in documents],
            "sparse_vector": sparse_embeddings,
            "dense_vector": dense_embeddings,
        }
        for f in self.document_cls.metadata_schema():
            insert_dict[f.name] = [self._get_first_field_value(doc, f.name) for doc in documents]

        self.operator.buffered_insert([insert_dict[k] for k in self.config.field_names()])

    def _get_first_field_value(self, doc: Document, name: str):
        for field in doc.metadata():
            if field.name == name:
                return field.contents[0] if field.contents else None
        return None

    def get_query(self, filter: Filter) -> str:
        clauses = []
        # Filter fields: field value is one of the filter options
        for field in self.filter_cls.filter_fields():
            values = getattr(filter, field, None)
            if values:
                clauses.append(f'{field} in ["' + '","'.join(map(str, values)) + '"]')

        # Must fields: treated the same way due to Milvus limitations
        for field in self.filter_cls.must_fields():
            values = getattr(filter, field, None)
            if values:
                clauses.append(f'{field} in ["' + '","'.join(map(str, values)) + '"]')

        return " and ".join(clauses) if clauses else None

    def search(self, query: str, filter: Filter, limit: int = 10) -> List[str]:
        dense_vector, sparse_vector = self.embed_query(query)
        results = self.operator.search_hybrid(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            alpha=0.5,
            limit=limit,
            output_fields=["pk"],
            expr=self.get_query(filter)
        )
        return [hit.fields.get("pk", "") for hit in results[0]]

    def spec(self) -> SearchSpec:
        return SearchSpec(name="milvus_search_engine", optimal_for="weak")


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
        return SearchSpec(name="sqlite_search_engine", optimal_for="strong")
    

class ElasticSearchEngine(SearchEngine):
    document_cls: Type[Document]
    filter_cls: Type[Filter]

    def __init__(
        self,
        es_host: str,
        document_cls: Type[Document],
        filter_cls: Type[Filter],
        es_index: str = "elastic_index"
    ):
        config = yaml.safe_load(open("config/elastic_search.yml", "r"))
        self.es = Elasticsearch(
            [es_host],
            basic_auth=("elastic", config["password"]),
            verify_certs=True,
            ca_certs=config["ca_certs"]
        )
        self.es_index = es_index
        self.document_cls = document_cls
        self.filter_cls = filter_cls

    def setup(self):
        if self.es.indices.exists(index=self.es_index):
            self.es.indices.delete(index=self.es_index)

        mapping = {
            "mappings": {
                "properties": {
                    f.name: {"type": "keyword" if f.type.value == "str" else "integer"}
                    for f in self.document_cls.metadata_schema()
                }
            }
        }
        self.es.indices.create(index=self.es_index, body=mapping)

    def insert(self, docs: List[Document]):
        for doc in docs:
            body = {
                f.name: f.contents
                for f in doc.metadata() 
            }
            self.es.index(index=self.es_index, id=doc.key(), body=body)

    def get_query(self, filter: Filter) -> Dict:
        must_clauses = []
        filter_clauses = []

        for field in self.filter_cls.must_fields():
            values = getattr(filter, field, None)
            if not values:
                continue
            for val in values:
                must_clauses.append({"match_phrase": {field: val}})

        for field in self.filter_cls.filter_fields():
            values = getattr(filter, field, None)
            if not values:
                continue
            filter_clauses.append({"terms": {field: values}})

        return {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses
                }
            }
        }

    def search(self, query: str, filter: Filter, limit: int = 10000) -> List[str]:
        es_query = self.get_query(filter)
        response = self.es.search(index=self.es_index, body=es_query, size=limit)
        return [hit["_id"] for hit in response["hits"]["hits"]]

    def spec(self) -> SearchSpec:
        return SearchSpec(name="elastic_search_engine", optimal_for="strong")