import os
from dotenv import load_dotenv
from loguru import logger

# 載入環境變數
load_dotenv()

class Config:
    """配置類別，用於管理所有設定"""
    
    # 資料路徑
    ABSTRACT_PATH = os.getenv("ABSTRACT_PATH")
    DATA_PATH = os.getenv("DATA_PATH")
    CONTENT_PATH = os.getenv("CONTENT_PATH")
    NUM_FILES = 2437  # 可以考慮移到環境變數
    
    # 資料庫設定
    DB_URI = os.getenv("DB_URI", "./milvus.db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hybrid_demo")
    
    # 搜尋設定
    DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", "3"))
    DEFAULT_SPARSE_WEIGHT = float(os.getenv("DEFAULT_SPARSE_WEIGHT", "1.0"))
    DEFAULT_DENSE_WEIGHT = float(os.getenv("DEFAULT_DENSE_WEIGHT", "1.0"))
    
    # LLM 設定
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:latest")
    
    @classmethod
    def validate(cls):
        """驗證配置是否正確"""
        required_paths = [cls.ABSTRACT_PATH, cls.DATA_PATH, cls.CONTENT_PATH]
        for path in required_paths:
            if not path:
                raise ValueError(f"Required path not set in environment variables")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path does not exist: {path}")
        
        logger.info("Configuration validated successfully")
    
    @classmethod
    def get_search_params(cls, sparse_weight=None, dense_weight=None):
        """獲取搜尋參數"""
        return {
            "sparse_weight": sparse_weight or cls.DEFAULT_SPARSE_WEIGHT,
            "dense_weight": dense_weight or cls.DEFAULT_DENSE_WEIGHT,
            "limit": cls.DEFAULT_SEARCH_LIMIT
        }

# 初始化時驗證配置
Config.validate()