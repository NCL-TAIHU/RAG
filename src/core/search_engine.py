from typing import List, Optional, Dict, Type, Tuple
from src.core.document import Document, FieldType
from src.core.filter import Filter
from src.core.embedder import BaseEmbedder, SparseEmbedder, DenseEmbedder
from src.core.collection import FieldConfig, IndexConfig, CollectionConfig, CollectionOperator, CollectionBuilder
from src.core.elastic import ElasticIndexBuilder, ElasticIndexConfig
from src.core.util import get 
from typing import List
from pymilvus import (
    DataType,
    Collection,
)
from pymilvus.client.abstract import Hits, Hit
from elasticsearch import Elasticsearch
from pydantic import BaseModel
import yaml
import logging
from src.core.vector_manager import VectorManager
from scipy.sparse import csr_array
from src.core.util import coalesce

logger = logging.getLogger('taihu')

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

class HybridMilvusSearchEngine(SearchEngine):
    document_cls: Type[Document]
    filter_cls: Type[Filter]

    def __init__(
        self,
        document_cls: Type[Document],
        filter_cls: Type[Filter],
        dense_vm: VectorManager, 
        sparse_vm: VectorManager, 
        alpha = 0.5, 
        force_rebuild: bool = False,
    ):
        self.document_cls = document_cls
        self.filter_cls = filter_cls
        self.dense_vm = dense_vm
        self.sparse_vm = sparse_vm
        self.alpha = alpha
        self.force_rebuild = force_rebuild

        assert self.dense_vm.get_vs_metadata().dataset == self.sparse_vm.get_vs_metadata().dataset, \
            "Dense and sparse vector managers must have the same dataset"

        assert self.dense_vm.get_vs_metadata().channel == self.sparse_vm.get_vs_metadata().channel, \
            "Dense and sparse vector managers must have the same channel"
        
        assert self.dense_vm.get_vs_metadata().chunker_meta == self.sparse_vm.get_vs_metadata().chunker_meta, \
            "Dense and sparse vector managers must have the same chunker metadata"

        fields = [
            FieldConfig(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100)
        ]

        vs_meta = dense_vm.get_vs_metadata()
        dense_model = vs_meta.model
        sparse_model = sparse_vm.get_vs_metadata().model
        dataset = vs_meta.dataset
        channel = vs_meta.channel
        chunker_type = vs_meta.chunker_meta.chunker_type

        self.collection_name = f"{dense_model}_{sparse_model}_{dataset}_{channel}_{chunker_type}_hybrid_collection"


        metadata_schema = self.document_cls.metadata_schema()
        for field_name in self.filter_cls.filter_fields():
            f = metadata_schema[field_name]
            fields.append(FieldConfig(
                name=f.name,
                dtype=f.type.to_milvus_type(),
                max_length=f.max_len
            ))

        fields += [
            FieldConfig(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldConfig(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_vm.embedder.get_dim())
        ]

        self.config = CollectionConfig(
            collection_name=self.collection_name,
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
        logger.info(f"Setting up Milvus collection: {self.config.collection_name}, force_rebuild={self.force_rebuild}")
        builder = CollectionBuilder.from_config(self.config)
        builder.connect()
        self.collection = (builder.build() 
                           if self.force_rebuild 
                           else coalesce(builder.get_existing, builder.build))

        self.operator = CollectionOperator(self.collection)

    def embed_query(self, query: str):
        dense = self.dense_vm.get_raw_embedding([query])[0]
        sparse = self.sparse_vm.get_raw_embedding([query])
        assert sparse.shape[0] == 1, "Expected a single-row sparse vector"
        return dense, sparse._getrow(0)

    def insert(self, documents: List[Document]):
        existing_pks = set()
        if documents:
            keys = [doc.key() for doc in documents]
            expr = f'pk in ["{"","".join(keys)}"]'
            self.collection.load()
            results = self.collection.query(expr, output_fields=["pk"])
            existing_pks = {res["pk"] for res in results}

        new_docs = [doc for doc in documents if doc.key() not in existing_pks]
        if not new_docs:
            return

        dense_embeddings = self.dense_vm.get_doc_embeddings(new_docs)  # Dict[str, List[List[float]]]
        sparse_embeddings = self.sparse_vm.get_doc_embeddings(new_docs)  # Dict[str, csr_array]

        insert_dict = {
            "pk": [],
            "dense_vector": [],
            "sparse_vector": [],
        }

        for doc in new_docs:
            doc_id = doc.key()
            dense_chunks = dense_embeddings[doc_id]
            sparse_chunks = sparse_embeddings[doc_id]
            assert len(dense_chunks) == sparse_chunks.shape[0], \
                  f"Mismatch in chunk counts for {doc_id}. This should be checked at initialization. Same channel and chunker should yield same number of chunks."

            for i, (dense_vec, sparse_vec) in enumerate(zip(dense_chunks, sparse_chunks)):
                chunk_id = f"{doc_id}-{i}"
                insert_dict["pk"].append(chunk_id)
                insert_dict["dense_vector"].append(dense_vec)
                insert_dict["sparse_vector"].append(sparse_vec)

        metadatas = [doc.metadata() for doc in new_docs]
        for f_name in self.filter_cls.filter_fields():
            values = []
            for i, doc in enumerate(new_docs):
                val = get(metadatas[i][f_name].contents, 0, metadatas[i][f_name].type.default_value())
                values.extend([val] * len(dense_embeddings[doc.key()]))
            insert_dict[f_name] = values

        self.operator.buffered_insert([insert_dict[k] for k in self.config.field_names()])

    def _get_query(self, filter: Filter) -> Optional[str]:
        clauses = []
        schema_map = filter._doc_cls_.metadata_schema()
        logger.debug("milvus does not support must fields, only filter fields")
        for field in self.filter_cls.filter_fields():
            values = getattr(filter, field, None)
            if values:
                field_type = schema_map[field].type
                if field_type in {FieldType.STRING, FieldType.BOOLEAN}:
                    formatted = ",".join(f'"{v}"' for v in values)
                else:
                    formatted = ",".join(f"{v}" for v in values)
                clauses.append(f"{field} in [{formatted}]")

        return " and ".join(clauses) if clauses else None

    def _grouped(self, chunk_ids: List[str]) -> List[str]: 
        grouped = set()
        for cid in chunk_ids:
            doc_id = cid.split("-")[0]
            grouped.add(doc_id)
        return list(grouped)

    def search(self, query: str, filter: Filter, limit: int = 100) -> List[str]:
        dense_vector, sparse_vector = self.embed_query(query)
        expr = self._get_query(filter)
        results = self.operator.search_hybrid(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            alpha=self.alpha,
            limit=limit,
            output_fields=["pk"],
            expr=expr
        )
        return self._grouped([hit.fields.get("pk", "") for hit in results[0]])

    def spec(self) -> SearchSpec:
        return SearchSpec(name="milvus_search_engine", optimal_for="weak")


class MilvusSearchEngine(SearchEngine):
    document_cls: Type[Document]
    filter_cls: Type[Filter]

    def __init__(
        self,
        vector_type: str,
        vector_manager: VectorManager,
        document_cls: Type[Document],
        filter_cls: Type[Filter],
        force_rebuild: bool = False,
    ):
        self.vector_type = vector_type  # "dense" or "sparse"
        self.vm = vector_manager
        self.document_cls = document_cls
        self.filter_cls = filter_cls
        self.force_rebuild = force_rebuild

        vs_meta = vector_manager.get_vs_metadata()
        model = vs_meta.model
        dataset = vs_meta.dataset
        channel = vs_meta.channel
        chunker_type = vs_meta.chunker_meta.chunker_type
        self.collection_name = f"{model}_{dataset}_{channel}_{chunker_type}_{vector_type}_collection"

        fields = [
            FieldConfig(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100)
        ]
        metadata_schema = self.document_cls.metadata_schema()
        for field_name in self.filter_cls.filter_fields():
            f = metadata_schema[field_name]
            fields.append(FieldConfig(
                name=f.name,
                dtype=f.type.to_milvus_type(),
                max_length=f.max_len
            ))

        if self.vector_type == "sparse":
            fields.append(FieldConfig(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR))
            indexes = [
                IndexConfig(
                    field_name="sparse_vector",
                    index_params={"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
                )
            ]
        elif self.vector_type == "dense":
            fields.append(FieldConfig(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedder.get_dim()))
            indexes = [
                IndexConfig(
                    field_name="dense_vector",
                    index_params={
                        "index_type": "IVF_FLAT",
                        "metric_type": "IP",
                        "params": {"nlist": 128}
                    }
                )
            ]
        else:
            raise ValueError("Unsupported vector_type. Must be 'dense' or 'sparse'.")

        self.config = CollectionConfig(
            collection_name=self.collection_name,
            fields=fields,
            indexes=indexes
        )

    def setup(self):
        logger.info(f"Setting up Milvus collection: {self.config.collection_name}, force_rebuild={self.force_rebuild}")
        builder = CollectionBuilder.from_config(self.config)
        builder.connect()
        self.collection = (builder.build() 
                           if self.force_rebuild 
                           else coalesce(builder.get_existing, builder.build))

        self.operator = CollectionOperator(self.collection)

    def embed_query(self, query: str):
        embedding = self.vm.get_raw_embedding([query])
        if self.vector_type == "dense":
            assert isinstance(embedding, List) and len(embedding) == 1, "Dense embedder must return a single vector"
            return embedding[0]
        elif self.vector_type == "sparse":
            assert isinstance(embedding, csr_array), "Sparse embedder must return csr_array"
            assert embedding.shape[0] == 1, "Expected a single-row sparse vector"
            return embedding._getrow(0)

    def insert(self, documents: List[Document]):
        existing_pks = set()
        if documents:
            keys = [doc.key() for doc in documents]
            expr = f'pk in ["{"","".join(keys)}"]'
            self.collection.load()
            results = self.collection.query(expr, output_fields=["pk"])
            existing_pks = {res["pk"] for res in results}

        new_docs = [doc for doc in documents if doc.key() not in existing_pks]
        if not new_docs:
            return

        embeddings = self.vm.get_doc_embeddings(new_docs)
        insert_dict = {
            "pk": [],
            self.vector_type + "_vector": [],
        }

        for doc in new_docs:
            doc_id = doc.key()
            chunks = embeddings[doc_id]
            for i, emb in enumerate(chunks):
                chunk_id = f"{doc_id}-{i}"
                insert_dict["pk"].append(chunk_id)
                insert_dict[self.vector_type + "_vector"].append(emb)

        metadatas = [doc.metadata() for doc in new_docs]
        for f_name in self.filter_cls.filter_fields():
            values = []
            for i, doc in enumerate(new_docs):
                val = get(metadatas[i][f_name].contents, 0, metadatas[i][f_name].type.default_value())
                values.extend([val] * len(embeddings[doc.key()]))
            insert_dict[f_name] = values

        self.operator.buffered_insert([insert_dict[k] for k in self.config.field_names()])

    def _get_query(self, filter: Filter) -> Optional[str]:
        clauses = []
        schema_map = filter._doc_cls_.metadata_schema()
        logger.debug("milvus does not support must fields, only filter fields")
        for field in self.filter_cls.filter_fields():
            values = getattr(filter, field, None)
            if values:
                field_type = schema_map[field].type
                if field_type in {FieldType.STRING, FieldType.BOOLEAN}:
                    formatted = ",".join(f'"{v}"' for v in values)
                else:
                    formatted = ",".join(f"{v}" for v in values)
                clauses.append(f"{field} in [{formatted}]")

        return " and ".join(clauses) if clauses else None

    def _grouped(self, chunk_ids: List[str]) -> List[str]: 
        grouped = set()
        for cid in chunk_ids:
            doc_id = cid.split("-")[0]
            grouped.add(doc_id)
        return list(grouped)

    def search(self, query: str, filter: Filter, limit: int = 100) -> List[str]:
        vector = self.embed_query(query)
        expr = self._get_query(filter)
        results = self.operator.search(
            data=vector,
            anns_field=self.vector_type + "_vector",
            param={},
            limit=limit,
            expr=expr,
            output_fields=["pk"]
        )
        return self._grouped([hit.fields.get("pk", "") for hit in results[0]])

    def spec(self) -> SearchSpec:
        return SearchSpec(name="milvus_search_engine", optimal_for="weak")


class ElasticSearchEngine(SearchEngine):
    document_cls: Type[Document]
    filter_cls: Type[Filter]

    def __init__(
        self,
        es_host: str,
        document_cls: Type[Document],
        filter_cls: Type[Filter],
        es_index: str = "elastic_index",
        force_rebuild: bool = False
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
        self.force_rebuild = force_rebuild

        # Extract schema fields
        schema = self.document_cls.metadata_schema()
        self.field_types = {
            schema[f].name: "keyword" if schema[f].type.value == "str" else "integer"
            for f in self.filter_cls.filter_fields() + self.filter_cls.must_fields()
        }

    def setup(self): 
        # Build index if needed
        builder = ElasticIndexBuilder(
            es=self.es,
            config=ElasticIndexConfig(es_index=self.es_index, fields=self.field_types)
        )
        builder.build(force_rebuild=self.force_rebuild)

    def insert(self, docs: List[Document]):
        for doc in docs:
            if self.es.exists(index=self.es_index, id=doc.key()):
                continue  # Skip duplicates
            data = doc.metadata()
            body = {
                data[f].name: data[f].contents
                for f in self.filter_cls.filter_fields() + self.filter_cls.must_fields()
            }
            self.es.index(index=self.es_index, id=doc.key(), body=body)

    def _get_query(self, filter: Filter) -> Dict:
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
        es_query = self._get_query(filter)
        response = self.es.search(index=self.es_index, body=es_query, size=limit)
        return [hit["_id"] for hit in response["hits"]["hits"]]

    def spec(self) -> SearchSpec:
        return SearchSpec(name="elastic_search_engine", optimal_for="strong")