from typing import Dict, List, Any
from loguru import logger
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from src.utils.config import Config

class DatabaseManager:
    """資料庫管理類別，負責資料庫的操作"""
    
    def __init__(self):
        """初始化資料庫管理器"""
        self.collection = None
    
    def connect(self) -> None:
        """連接到 Milvus 資料庫"""
        try:
            connections.connect(uri=Config.DB_URI)
            logger.info(f"Connected to Milvus database at {Config.DB_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def setup_database(self, metadata: Dict[str, List[str]], abstract: List[str], 
                      docs_embeddings: Dict[str, Any], dense_dim: int) -> Collection:
        """設定 Milvus 資料庫"""
        try:
            self.connect()
            
            # 定義資料結構
            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
                FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=20000),
            ]
            schema = CollectionSchema(fields)
            
            # 建立集合
            if utility.has_collection(Config.COLLECTION_NAME):
                logger.info(f"Dropping existing collection: {Config.COLLECTION_NAME}")
                Collection(Config.COLLECTION_NAME).drop()
            
            logger.info(f"Creating new collection: {Config.COLLECTION_NAME}")
            self.collection = Collection(Config.COLLECTION_NAME, schema, consistency_level="Strong")
            
            # 建立索引
            logger.info("Creating indexes")
            sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
            self.collection.create_index("sparse_vector", sparse_index)
            dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
            self.collection.create_index("dense_vector", dense_index)
            self.collection.load()
            
            # 批次插入資料
            logger.info("Starting batch insertion of data")
            for i in range(0, len(abstract), 50):
                batched_entities = [
                    metadata['id'][i : i + 50],
                    abstract[i : i + 50],
                    docs_embeddings["sparse"][i : i + 50],
                    docs_embeddings["dense"][i : i + 50],
                    metadata['data'][i : i + 50],
                    metadata['content'][i : i + 50],
                ]
                self.collection.insert(batched_entities)
            
            logger.info(f"Successfully inserted {self.collection.num_entities} entities")
            return self.collection
            
        except Exception as e:
            logger.error(f"Error in setup_database: {str(e)}")
            raise
    
    def get_collection(self) -> Collection:
        """獲取已存在的集合"""
        try:
            self.connect()
            self.collection = Collection(Config.COLLECTION_NAME)
            self.collection.load()
            logger.info(f"Successfully loaded collection: {Config.COLLECTION_NAME}")
            return self.collection
        except Exception as e:
            logger.error(f"Error in get_collection: {str(e)}")
            raise 