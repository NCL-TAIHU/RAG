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
    def __init__(self, sparse_embedder: SparseEmbedder, dense_embedder: DenseEmbedder):
        self.sparse_embedder = sparse_embedder
        self.dense_embedder = dense_embedder
        self.config = CollectionConfig(
            collection_name="example_collection",
            fields=[
                FieldConfig(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldConfig(name="year", dtype=DataType.INT64),
                FieldConfig(name="category", dtype=DataType.VARCHAR, max_length=100),
                FieldConfig(name="school_chinese", dtype=DataType.VARCHAR, max_length=100),
                FieldConfig(name="school_english", dtype=DataType.VARCHAR, max_length=100),
                FieldConfig(name="dept_chinese", dtype=DataType.VARCHAR, max_length=100),
                FieldConfig(name="dept_english", dtype=DataType.VARCHAR, max_length=100),
                FieldConfig(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldConfig(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_embedder.get_dim()),
            ],
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
        dense_vector = self.dense_embedder.embed([query])[0]
        sparse_vector = self.sparse_embedder.embed([query])
        assert sparse_vector.shape[0] == 1, "Expected a single-row sparse vector"
        return dense_vector, sparse_vector._getrow(0)

    def insert(self, documents: List[Document]):
        documents = [doc for doc in documents if doc.chinese.abstract]
        ids = [doc.id for doc in documents]
        abstracts = [doc.chinese.abstract for doc in documents]
        dense_embeddings = self.dense_embedder.embed(abstracts)
        sparse_embeddings = self.sparse_embedder.embed(abstracts)

        years = [doc.year or -1 for doc in documents]
        categories = [doc.category or "" for doc in documents]
        school_chinese = [doc.chinese.school or "" for doc in documents]
        school_english = [doc.english.school or "" for doc in documents]
        dept_chinese = [doc.chinese.dept or "" for doc in documents]
        dept_english = [doc.english.dept or "" for doc in documents]

        self.operator.buffered_insert([
            ids, years, categories,
            school_chinese, school_english,
            dept_chinese, dept_english,
            sparse_embeddings, dense_embeddings
        ])

    def get_query(self, filter: Filter) -> Optional[str]: 
        expr_clauses = []

        if filter.ids:
            expr_clauses.append(f'pk in ["{"\",\"".join(filter.ids)}"]')
        if filter.years:
            expr_clauses.append(f'year in [{",".join(map(str, filter.years))}]')
        if filter.categories:
            expr_clauses.append(f'category in ["{"\",\"".join(filter.categories)}"]')
        if filter.schools:
            school_clause = (
                f'school_chinese in ["{"\",\"".join(filter.schools)}"] '
                f'or school_english in ["{"\",\"".join(filter.schools)}"]'
            )
            expr_clauses.append(f'({school_clause})')
        if filter.depts:
            dept_clause = (
                f'dept_chinese in ["{"\",\"".join(filter.depts)}"] '
                f'or dept_english in ["{"\",\"".join(filter.depts)}"]'
            )
            expr_clauses.append(f'({dept_clause})')

        return " and ".join(expr_clauses) if expr_clauses else None

    def search(self, query: str, filter: Filter, limit: int = 10) -> List[str]:
        dense_vector, sparse_vector = self.embed_query(query)
        results = self.operator.search_hybrid(
            dense_vector, 
            sparse_vector, 
            alpha=0.5, 
            limit=limit, 
            output_fields=["pk"],
            expr=self.get_query(filter)
        )
        hits: Hits = results[0]
        return [hit.fields.get("pk", "") for hit in hits]

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
    def __init__(self, es_host: str, es_index: str):
        config = yaml.safe_load(open("config/elastic_search.yml", "r"))
        self.es = Elasticsearch([es_host], 
                                basic_auth=("elastic", config["password"]),
                                verify_certs=True,
                                ca_certs=config["ca_certs"])
        self.es_index = es_index

    def setup(self):
        if self.es.indices.exists(index=self.es_index):
            self.es.indices.delete(index=self.es_index)

        mapping = {
            "mappings": {
                "properties": {
                    "year": {"type": "integer"},
                    "category": {"type": "keyword"},
                    "keywords": {"type": "keyword"},
                    "english.authors": {"type": "keyword"},
                    "chinese.authors": {"type": "keyword"},
                    "english.advisors": {"type": "keyword"},
                    "chinese.advisors": {"type": "keyword"}, 
                    "english.school": {"type": "keyword"},
                    "chinese.school": {"type": "keyword"},
                    "english.dept": {"type": "keyword"},
                    "chinese.dept": {"type": "keyword"}
                }
            }
        }

        self.es.indices.create(index=self.es_index, body=mapping)

    def insert(self, docs: List[Document]):
        for doc in docs:
            self.es.index(index=self.es_index, id=doc.id, body=doc.model_dump())

    def get_query(self, filter: Filter) -> Dict: 
        must_clauses = []
        filter_clauses = []

        # Must match all keywords
        for kw in filter.keywords:
            must_clauses.append({"match_phrase": {"keywords": kw}})

        
        # Must match each author (in either language)
        for author in filter.authors:
            must_clauses.append({
                "bool": {
                    "should": [
                        {"term": {"english.authors": author}},
                        {"term": {"chinese.authors": author}}
                    ],
                    "minimum_should_match": 1
                }
            })

        # Must match each advisor (in either language)
        for advisor in filter.advisors:
            must_clauses.append({
                "bool": {
                    "should": [
                        {"term": {"english.advisors": advisor}},
                        {"term": {"chinese.advisors": advisor}}
                    ],
                    "minimum_should_match": 1
                }
            })

        # Filter by IDs (exact match)
        if filter.ids:
            filter_clauses.append({"terms": {"_id": filter.ids}})

        # Filter by years (exact match)
        if filter.years:
            filter_clauses.append({"terms": {"year": filter.years}})

        # Filter by categories
        if filter.categories:
            filter_clauses.append({"terms": {"category": filter.categories}})

        # Filter by schools
        if filter.schools:
            filter_clauses.append({
                "bool": {
                    "should": [
                        {"terms": {"english.school": filter.schools}},
                        {"terms": {"chinese.school": filter.schools}}
                    ],
                    "minimum_should_match": 1
                }
            })

        # Filter by departments
        if filter.depts:
            filter_clauses.append({
                "bool": {
                    "should": [
                        {"terms": {"english.dept": filter.depts}},
                        {"terms": {"chinese.dept": filter.depts}}
                    ],
                    "minimum_should_match": 1
                }
            })

        es_query = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses
                }
            }
        }

        return es_query
    
    def search(self, query: str, filter: Filter, limit: int = 10000) -> List[str]:
        # # es_query = self.get_query(filter, limit)
        # es_query = self.get_query(filter)    # Here have some adjustment to solve some problems
        #  # Here have some adjustment to solve some problems
        # count = self.es.count(index=self.es_index, body=es_query)
        # response = self.es.search(index=self.es_index, body=es_query, size=min(count, limit))  
        
        # return [hit["_id"] for hit in response["hits"]["hits"]]
        es_query = self.get_query(filter)
        try:
            # 直接使用 search 查詢，不需要先計算總數
            response = self.es.search(
                index=self.es_index,
                body=es_query,
                size=limit
            )
            
            # 確保 response 是字典類型
            if isinstance(response, dict) and "hits" in response and "hits" in response["hits"]:
                return [hit["_id"] for hit in response["hits"]["hits"]]
            else:
                print(f"Unexpected Elasticsearch response format: {response}")
                return []
        except Exception as e:
            print(f"Elasticsearch search error: {str(e)}")
            return []        

    def spec(self) -> SearchSpec:
        return SearchSpec(name="elastic_search_engine", optimal_for="strong")