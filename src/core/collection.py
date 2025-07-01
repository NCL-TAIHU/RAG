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
from typing import List, Dict, Optional
from scipy.sparse import csr_array
import logging

logger = logging.getLogger(__name__)

class FieldConfig(BaseModel): 
    name: str
    dtype: DataType
    is_primary: bool = False
    max_length: int = 10000
    dim: int = None  
    is_partition_key: bool = False
    default_value: str = None

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
    builds a Milvus collection specifying the collection fields and indexes
    '''
    collection_name: str
    fields: list[FieldConfig] = field(default_factory=list)
    indexes: list[IndexConfig] = field(default_factory=list)
    consistency_level: str = "Strong"   


    @classmethod
    def from_config(cls, config: CollectionConfig):
        return cls(**config.__dict__)  
        
    def get_config(self) -> CollectionConfig:
        return CollectionConfig(**asdict(self))
    
    def connect(self):
        connections.connect(uri="db/milvus.db")  # Adjust the URI as needed

    def get_existing(self) -> Collection:
        '''
        Returns the existing collection instance. 
        Requires: Collection already exists.
        '''
        return Collection(self.collection_name)
    
    def build(self) -> Collection:
        #drop existing collection if it exists
        if utility.has_collection(self.collection_name):
            Collection(self.collection_name).drop()

        field_schemas = [FieldSchema(**field.model_dump(exclude_none=True)) for field in self.fields]
        collection = Collection(self.collection_name, CollectionSchema(field_schemas), consistency_level = self.consistency_level)
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
    
    def search_dense(
            self, 
            query_vector: List[float], 
            limit: int = 10, 
            output_fields: List[str] = ["pk"], 
            expr: Optional[str] = None
        ):
        '''
        Performs a dense vector search in the collection.
        query_vector: List[float], a dense vector to search for.
        limit: int, the maximum number of results to return.
        output_fields: List[str], fields to return in the results.
        expr: Optional[str], an expression to filter the results.
        '''
        self.collection.load()
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param=search_params,
            limit=limit,
            expr=expr, 
            output_fields=output_fields
        )
        return results[0] if results else []
    
    def search_sparse(
            self, 
            query_vector: csr_array, 
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
            data=query_vector,
            anns_field="sparse_vector",
            param=search_params,
            limit=limit,
            expr=expr, 
            output_fields=output_fields, 
        )
        return results[0] if results else []

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