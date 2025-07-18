from dataclasses import dataclass, asdict, field
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from pymilvus.client.abstract import SearchResult
from pydantic import BaseModel
from tqdm import tqdm
from typing import List, Dict, Optional, Union
from scipy.sparse import csr_array
import os
import json
import logging

logger = logging.getLogger('taihu')

class FieldConfig(BaseModel): 
    name: str
    dtype: DataType
    is_primary: bool = False
    max_length: int = 10000
    dim: Optional[int] = None  
    is_partition_key: bool = False
    default_value: Optional[str] = None

class IndexConfig(BaseModel):
    field_name: str
    index_params: dict = None

class CollectionConfig(BaseModel):
    '''
    Defines how to build a Milvus collection, uniquely determines ways to embed and retrieve from the db
    '''
    collection_name: str
    fields: list[FieldConfig] = []
    indexes: list[IndexConfig] = []
    consistency_level: str = "Strong"  # Default consistency level
    def get_field(self, name: str) -> FieldConfig:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Field '{name}' not found in collection schema.")

    def has_field(self, name: str) -> bool:
        return any(field.name == name for field in self.fields)

    def field_names(self) -> List[str]:
        return [field.name for field in self.fields]

    def primary_field(self) -> FieldConfig:
        for field in self.fields:
            if field.is_primary:
                return field
        raise ValueError("No primary key field defined in collection config.")


@dataclass
class CollectionBuilder:
    '''
    Builds a Milvus collection specifying the collection fields and indexes.
    '''
    collection_name: str
    fields: List[FieldConfig] = field(default_factory=list)
    indexes: List[IndexConfig] = field(default_factory=list)
    consistency_level: str = "Strong"
    config_path: str = field(init=False)

    def __post_init__(self):
        self.config_path = os.path.join("db", "collection_configs", f"{self.collection_name}.json")

    @classmethod
    def from_config(cls, config: CollectionConfig):
        return cls(**config.__dict__)  

    def get_config(self) -> CollectionConfig:
        return CollectionConfig(**asdict(self))

    def connect(self):
        connections.connect(uri="db/milvus.db")  # Adjust the URI as needed

    def get_existing(self) -> Optional[Collection]:
        '''
        Returns the existing collection instance.
        Requires: Collection already exists and matches the configuration of the builder.
        '''
        if not self._config_matches_saved(): 
            logger.info(f"Collection {self.collection_name} does not match saved config, cannot retrieve existing collection.")
            return None
        logger.info(f"Retrieving existing collection {self.collection_name} from Milvus.")
        return Collection(self.collection_name)

    def _config_matches_saved(self) -> bool:
        if not os.path.exists(self.config_path):
            return False
        with open(self.config_path, "r", encoding="utf-8") as f:
            logger.debug(f"Loading saved config from {self.config_path}")
            saved = CollectionConfig(**json.load(f))
        return saved.model_dump() == self.get_config().model_dump()

    def _save_config(self): 
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.get_config().model_dump(), f, indent=2)

    def build(self) -> Collection:
        # Drop existing collection if it exists
        logger.info(f"Building collection {self.collection_name} with fields: {[field.name for field in self.fields]} and indexes: {[index.field_name for index in self.indexes]}")
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()

        self._save_config()
        field_schemas = [FieldSchema(**field.model_dump(exclude_none=True)) for field in self.fields]
        schema = CollectionSchema(field_schemas)
        collection = Collection(self.collection_name, schema, consistency_level=self.consistency_level)
        for index in self.indexes:
            collection.create_index(**index.model_dump(exclude_none=True))
        return collection

class CollectionOperator:
    '''
    interacts with the Milvus database, ensures insertion and retrieval of data. 
    '''
    def __init__(self, collection: Collection):
        self.collection = collection
        self.buffer_size = 32  # Default buffer size for batch insertions

    def buffered_insert(self, data: list[list]):
        '''
        Inserts data into the collection in batches.
        '''
        for i, field in enumerate(data): 
            if field is None: 
                print(f"Field {i} is None")
            
        for i in range(0, len(data[0]), self.buffer_size):
            batch_data = [field[i:i + self.buffer_size] for field in data]
            self.collection.insert(batch_data)
    
    def search(
            self, 
            query_vector: Union[csr_array, List[float]], 
            anns_field: str, #dense_vector or sparse_vector
            limit: int = 10, 
            output_fields: List[str] = ["pk"], 
            expr: Optional[str] = None
        ):
        '''
        Performs a sparse vector search in the collection.
        query_vector: csr_array, a sparse vector to search for.
        limit: int, the maximum number of results to return.
        output_fields: List[str], fields to return in the results.
        expr: Optional[str], an expression to filter the results.
        '''
        self.collection.load()
        search_params = {"metric_type": "IP", "params": {}}
        results = self.collection.search(
            data=[query_vector],
            anns_field=anns_field,
            param=search_params,
            limit=limit,
            expr=expr, 
            output_fields=output_fields, 
        )
        return results

    def search_hybrid(
        self,
        dense_vector: List[float],
        sparse_vector: csr_array,
        alpha: float = 0.5,
        limit: int = 10,
        output_fields: List[str] = ["pk"], 
        expr: Optional[str] = None,
    ) -> SearchResult:
        """
        Performs a hybrid search with dense and sparse vectors.
        alpha ∈ [0, 1]: higher = more weight on dense similarity.
        """
        self.collection.load()
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [dense_vector], "dense_vector", dense_search_params, limit=limit, expr=expr
        )
        #print(f"dense vector size {len(dense_vector)}")

        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [sparse_vector], "sparse_vector", sparse_search_params, limit=limit, expr=expr
        )
        #print(f"sparse vector size {sparse_vector.shape[1]}")
        search_params = {
            "metric_type": "IP",
            "params": {
                "alpha": alpha,  # balance between dense and sparse
                "hybrid": True
            }
        }
        # 使用 WeightedRanker 進行混合搜尋
        rerank = WeightedRanker(alpha, 1 - alpha)
        results = self.collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=output_fields,
        )
        return results