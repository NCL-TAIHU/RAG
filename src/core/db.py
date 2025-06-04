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

class FieldConfig(BaseModel): 
    name: str
    dtype: DataType
    is_primary: bool = False
    max_length: int = 10000
    dim: int = None  # Only for vector fields
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
    

class CollectionManager:
    '''
    interacts with the Milvus database, ensures insertion and retrieval of data. 
    '''
    def __init__(self, collection: Collection):
        self.collection = collection
        self.buffer_size = 32  # Default buffer size for batch insertions

    def buffered_insert(self, data: list[list]):
        #all the lists in data should have the same length
        for i, field in enumerate(data): 
            if field is None: 
                print(f"Field {i} is None")
            
        for i in range(0, len(data[0]), self.buffer_size):
            batch_data = [field[i:i + self.buffer_size] for field in data]
            self.collection.insert(batch_data)

    def _subset_expr(self, subset_ids: List[str]) -> str:
            """
            Constructs an expression for filtering results based on a list of IDs.
            """
            if not subset_ids:
                return None
            # Escape double quotes in IDs and format them
            escaped_ids = [f'"{id_}"' for id_ in subset_ids]
            return f"pk in [{', '.join(escaped_ids)}]"
    
    def search_dense(
            self, 
            query_vector: List[float], 
            limit: int = 10, 
            subset_ids: Optional[List[str]] = None,
            output_fields: List[str] = ["pk", "abstract", "keywords", "content"]
        ):
        self.collection.load()
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param=search_params,
            limit=limit,
            expr=self._subset_expr(subset_ids),  # Filter by subset if provided
            output_fields=output_fields
        )
        return results[0] if results else []
    
    def search_sparse(
            self, 
            query_vector: csr_array, 
            limit: int = 10, 
            subset_ids: Optional[List[str]] = None,
            output_fields: List[str] = ["pk", "abstract", "keywords", "content"]
        ):
        self.collection.load()
        search_params = {"metric_type": "IP", "params": {}}
        results = self.collection.search(
            data=query_vector,
            anns_field="sparse_vector",
            param=search_params,
            limit=limit,
            expr=self._subset_expr(subset_ids),  # Filter by subset if provided
            output_fields=output_fields, 
        )
        return results[0] if results else []

    def search_hybrid(
        self,
        dense_vector: List[float],
        sparse_vector: csr_array,
        alpha: float = 0.5,
        limit: int = 10,
        subset_ids: Optional[List[str]] = None,
        output_fields: List[str] = ["pk", "abstract", "keywords", "content"]
    ) -> SearchResult:
        """
        Performs a hybrid search with dense and sparse vectors.
        alpha ∈ [0, 1]: higher = more weight on dense similarity.
        """
        self.collection.load()
        dense_search_params = {"metric_type": "IP", "params": {}}
        dense_req = AnnSearchRequest(
            [dense_vector], "dense_vector", dense_search_params, limit=limit, expr=self._subset_expr(subset_ids)
        )
        print(f"dense vector size {len(dense_vector)}")

        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            [sparse_vector], "sparse_vector", sparse_search_params, limit=limit, expr=self._subset_expr(subset_ids)
        )
        print(f"sparse vector size {sparse_vector.shape[1]}")
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