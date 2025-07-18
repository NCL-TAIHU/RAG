from typing import List, Optional, Dict, Type, Tuple
from src.core.document import Document, FieldType
from src.core.filter import Filter
from src.core.embedder import BaseEmbedder, SparseEmbedder, DenseEmbedder
from src.core.collection import FieldConfig, IndexConfig, CollectionConfig, CollectionOperator, CollectionBuilder
from src.core.elastic import ElasticIndexBuilder, ElasticIndexConfig
from src.core.vector_set import BaseVectorSet
from src.core.schema import SearchEngineConfig, MilvusConfig, HybridMilvusConfig, ElasticSearchConfig, SequentialConfig
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
from scipy.sparse import csr_array, vstack
from src.core.util import coalesce

logger = logging.getLogger('taihu')
model_config = yaml.safe_load(open("config/model.yml", "r"))
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
    
    @classmethod
    def from_config(cls, config: SearchEngineConfig) -> 'SearchEngine':
        """
        Factory method to create a SearchEngine instance from a configuration.
        :param config: Configuration object containing search engine parameters.
        :return: An instance of SearchEngine.
        """
        if isinstance(config, MilvusConfig):
            return MilvusSearchEngine.from_config(config)
        elif isinstance(config, HybridMilvusConfig):
            return HybridMilvusSearchEngine.from_config(config)
        elif isinstance(config, ElasticSearchConfig):
            return ElasticSearchEngine.from_config(config)
        elif isinstance(config, SequentialConfig):
            return Sequential.from_config(config)
        else:
            raise ValueError(f"Unknown search engine type: {config.type}. Supported types: 'milvus', 'hybrid_milvus', 'elastic_search', 'sequential'.")
        
    def spec(self) -> SearchSpec: 
        raise NotImplementedError("This method should be overridden by subclasses.")

class Sequential(SearchEngine):
    """
    A compositional search engine where each engine refines the result of the previous.
    The first engine runs unconstrained; all others receive a filtered ID set.
    """
    def __init__(self, engines: List[SearchEngine]):
        assert len(engines) >= 2, "SequentialSearchEngine needs at least two engines"
        self.engines = engines

    @classmethod
    def from_config(cls, config: SequentialConfig) -> 'Sequential':
        """
        Factory method to create a Sequential instance from a configuration.
        :param config: Configuration object containing sequential search engine parameters.
        :return: An instance of Sequential.
        """
        engines = [SearchEngine.from_config(engine_cfg) for engine_cfg in config.engines]
        return cls(engines=engines)

    def setup(self):
        for engine in self.engines:
            engine.setup()

    def insert(self, docs: List[Document]):
        for engine in self.engines:
            engine.insert(docs)

    def search(self, query: str, filter: Filter, limit: int = 10) -> List[str]:
        current_filter = filter
        for i, engine in enumerate(self.engines):
            results = engine.search(query, current_filter, limit=limit)
            # Feed filtered results to the next stage
            if i < len(self.engines) - 1:
                current_filter = filter.model_copy(update={"ids": results})
        return results
    

    def spec(self) -> SearchSpec:
        names = " → ".join(e.spec().name for e in self.engines)
        return SearchSpec(name=f"sequential({names})", optimal_for="cascaded")


class HybridMilvusSearchEngine(SearchEngine):
    document_cls: Type[Document]
    filter_cls: Type[Filter]

    def __init__(
        self,
        dense_vector_set: BaseVectorSet, 
        sparse_vector_set: BaseVectorSet, 
        alpha = 0.5, 
        force_rebuild: bool = False,
    ):
        dataset = dense_vector_set.config() 
        self.document_cls = Document.from_dataset(dataset)
        self.filter_cls = Filter.from_dataset(dataset)
        self.dense_vector_set = dense_vector_set
        self.sparse_vector_set = sparse_vector_set
        self.alpha = alpha
        self.force_rebuild = force_rebuild
        assert self.dense_vector_set.config().embedder.embedding_type == "dense", \
            "Dense vector set must use a dense embedder"
        assert self.sparse_vector_set.config().embedder.embedding_type == "sparse", \
            "Sparse vector set must use a sparse embedder"
        
        self.dense_embedder: DenseEmbedder = BaseEmbedder.from_config(self.dense_vector_set.config().embedder)
        self.sparse_embedder: SparseEmbedder = BaseEmbedder.from_config(self.sparse_vector_set.config().embedder)

        assert self.dense_vector_set.config().dataset == self.sparse_vector_set.config().dataset, \
            "Dense and sparse vector managers must have the same dataset"

        assert self.dense_vector_set.config().channel == self.sparse_vector_set.config().channel, \
            "Dense and sparse vector managers must have the same channel"
        
        assert self.dense_vector_set.config().chunker == self.sparse_vector_set.config().chunker, \
            "Dense and sparse vector managers must have the same chunker metadata"

        fields = [
            FieldConfig(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100)
        ]

        vs_config = dense_vector_set.config()
        dense_model = vs_config.embedder.model_name
        sparse_model = sparse_vector_set.config().embedder.model_name
        dataset = vs_config.dataset
        channel = vs_config.channel

        #collection name has to be different enough so that collections don't collide. But even if collections of different 
        #config but same name do collide, the collection builder would handle it can build a new one. 
        self.collection_name = (f"dense={model_config[dense_model]['alias']}_\
                                sparse={model_config[sparse_model]['alias']}_\
                                dataset={dataset}_\
                                channel={channel}")


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
            FieldConfig(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_vector_set.embedder.get_dim())
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

    @classmethod
    def from_config(cls, config: HybridMilvusConfig) -> 'HybridMilvusSearchEngine':
        assert config.dense_vector_set.embedder.embedding_type == "dense", \
            "Dense vector set must use a dense embedder"
        assert config.sparse_vector_set.embedder.embedding_type == "sparse", \
            "Sparse vector set must use a sparse embedder"
        
        assert config.dense_vector_set.dataset == config.sparse_vector_set.dataset, \
            "Dense and sparse vector sets must have the same dataset"
        
        assert config.dense_vector_set.channel == config.sparse_vector_set.channel, \
            "Dense and sparse vector sets must have the same channel"
        assert config.dense_vector_set.chunker == config.sparse_vector_set.chunker, \
            "Dense and sparse vector sets must have the same chunker configuration"
        
        dense_vs = BaseVectorSet.from_config(config.dense_vector_set)
        sparse_vs = BaseVectorSet.from_config(config.sparse_vector_set)
        
        return cls(
            dense_vector_set=dense_vs,
            sparse_vector_set=sparse_vs,
            alpha=config.alpha,
        )
        

    def setup(self):
        logger.info(f"Setting up Milvus collection: {self.config.collection_name}, force_rebuild={self.force_rebuild}")
        builder = CollectionBuilder.from_config(self.config)
        builder.connect()
        self.collection = (builder.build() 
                           if self.force_rebuild 
                           else coalesce(builder.get_existing, builder.build))
        self.dense_vector_set.setup()
        self.sparse_vector_set.setup()
        self.operator = CollectionOperator(self.collection)

    def embed_query(self, query: str):
        dense = self.dense_embedder.embed([query])[0]
        sparse = self.sparse_embedder.embed([query])
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
        
        self.dense_vector_set.upsert([doc for doc in new_docs if not self.dense_vector_set.has(doc.key())])
        self.sparse_vector_set.upsert([doc for doc in new_docs if not self.sparse_vector_set.has(doc.key())])

        ids = [doc.key() for doc in new_docs]
        dense_embeddings = self.dense_vector_set.retrieve(ids)  # Dict[str, List[List[float]]]
        sparse_embeddings = self.sparse_vector_set.retrieve(ids)  # Dict[str, csr_array]

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

        insert_dict['sparse_vector'] = vstack(insert_dict['sparse_vector'])
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
        vector_set: BaseVectorSet,
        force_rebuild: bool = False,
    ):
        self.vector_set = vector_set
        self.dataset = vector_set.config().dataset
        self.vector_type = self.vector_set.config().embedder.embedding_type
        self.document_cls = Document.from_dataset(self.dataset)
        self.filter_cls = Filter.from_dataset(self.dataset)
        self.force_rebuild = force_rebuild
        self.embedder: BaseEmbedder = BaseEmbedder.from_config(self.vector_set.config().embedder)
        vs_config = vector_set.config()
        model = vs_config.embedder.model_name 
        dataset = vs_config.dataset
        channel = vs_config.channel
        self.collection_name = (f"{model_config[model]['alias']}_{dataset}_{channel}_collection")

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
            fields.append(FieldConfig(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_set.embedder.get_dim()))
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

    @classmethod
    def from_config(cls, config: MilvusConfig) -> 'MilvusSearchEngine':
        """
        Factory method to create a MilvusSearchEngine instance from a configuration.
        :param config: Configuration object containing Milvus search engine parameters.
        :return: An instance of MilvusSearchEngine.
        """
        return cls(vector_set = BaseVectorSet.from_config(config.vector_set)) 
    
    def setup(self):
        logger.info(f"Setting up Milvus collection: {self.config.collection_name}, force_rebuild={self.force_rebuild}")
        builder = CollectionBuilder.from_config(self.config)
        builder.connect()
        self.collection = (builder.build() 
                           if self.force_rebuild 
                           else coalesce(builder.get_existing, builder.build))
        self.vector_set.setup()
        self.operator = CollectionOperator(self.collection)

    def embed_query(self, query: str):
        embedding = self.embedder.embed([query])
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

        self.vector_set.upsert([doc for doc in new_docs if not self.vector_set.has(doc.key())])
        embeddings = self.vector_set.retrieve([doc.key() for doc in new_docs])
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
        dataset: str, 
        es_host: str,
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
        self.document_cls = Document.from_dataset(dataset)
        self.filter_cls = Filter.from_dataset(dataset)
        self.force_rebuild = force_rebuild

        # Extract schema fields
        schema = self.document_cls.metadata_schema()
        self.field_types = {
            schema[f].name: "keyword" if schema[f].type.value == "str" else "integer"
            for f in self.filter_cls.filter_fields() + self.filter_cls.must_fields()
        }

    @classmethod
    def from_config(cls, config: ElasticSearchConfig) -> 'ElasticSearchEngine':
        """
        Factory method to create an ElasticSearchEngine instance from a configuration.
        :param config: Configuration object containing ElasticSearch parameters.
        :return: An instance of ElasticSearchEngine.
        """
        return cls(
            dataset=config.dataset,
            es_host=config.es_host,
            es_index=config.es_index,
        )

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