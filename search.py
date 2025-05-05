from typing import Dict, Any, List, Optional
from loguru import logger
from pymilvus import AnnSearchRequest, WeightedRanker, Collection
from config import Config
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SearchEngine:
    """搜尋引擎類別，負責處理各種搜尋操作"""
    
    def __init__(self, collection: Collection):
        """初始化搜尋引擎
        
        Args:
            collection: 向量集合
        """
        self.collection = collection
        logger.info("SearchEngine initialized successfully")
    
    def dense_search(self, query_dense_embedding: Any, limit: int = None) -> List[Any]:
        """執行密集向量搜尋
        
        Args:
            query_dense_embedding: 查詢的密集向量
            limit: 結果數量限制
            
        Returns:
            List[Any]: 搜尋結果
        """
        try:
            logger.info("Starting dense search")
            if not isinstance(query_dense_embedding, (list, np.ndarray)):
                logger.error(f"Invalid query_dense_embedding type: {type(query_dense_embedding)}")
                return []
                
            if not self.collection:
                logger.error("Collection is not initialized")
                return []
                
            results = self.collection.query(
                query_embeddings=[query_dense_embedding],
                n_results=limit
            )
            
            if not results:
                logger.warning("No results found in dense search")
                return []
                
            logger.info(f"Dense search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in dense_search: {str(e)}", exc_info=True)
            return []
    
    def sparse_search(self, query_sparse_embedding: Any, limit: int = None) -> List[Any]:
        """執行稀疏向量搜尋
        
        Args:
            query_sparse_embedding: 查詢的稀疏向量
            limit: 結果數量限制
            
        Returns:
            List[Any]: 搜尋結果
        """
        try:
            logger.info("Starting sparse search")
            if not hasattr(query_sparse_embedding, '_getrow'):
                logger.error("Invalid query_sparse_embedding format")
                return []
                
            if not self.collection:
                logger.error("Collection is not initialized")
                return []
                
            results = self.collection.query(
                query_embeddings=[query_sparse_embedding],
                n_results=limit
            )
            
            if not results:
                logger.warning("No results found in sparse search")
                return []
                
            logger.info(f"Sparse search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in sparse_search: {str(e)}", exc_info=True)
            return []
    
    def hybrid_search(
        self,
        query_dense_embedding: Any,
        query_sparse_embedding: Any,
        sparse_weight: float = None,
        dense_weight: float = None,
        limit: int = None
    ) -> List[Any]:
        """執行混合搜尋
        
        Args:
            query_dense_embedding: 查詢的密集向量
            query_sparse_embedding: 查詢的稀疏向量
            sparse_weight: 稀疏向量權重
            dense_weight: 密集向量權重
            limit: 結果數量限制
            
        Returns:
            List[Any]: 搜尋結果
        """
        try:
            logger.info("Starting hybrid search")
            
            # 參數檢查
            if not isinstance(query_dense_embedding, (list, np.ndarray)):
                logger.error(f"Invalid query_dense_embedding type: {type(query_dense_embedding)}")
                return []
                
            if not hasattr(query_sparse_embedding, '_getrow'):
                logger.error("Invalid query_sparse_embedding format")
                return []
                
            if not self.collection:
                logger.error("Collection is not initialized")
                return []
                
            # 設置權重
            if sparse_weight is None:
                sparse_weight = 0.5
            if dense_weight is None:
                dense_weight = 0.5
                
            logger.info(f"Using weights - sparse: {sparse_weight}, dense: {dense_weight}")
            
            # 創建搜尋請求
            dense_search_params = {"metric_type": "IP", "params": {}}
            dense_req = AnnSearchRequest(
                [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
            )
            
            sparse_search_params = {"metric_type": "IP", "params": {}}
            sparse_req = AnnSearchRequest(
                [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
            )
            
            # 使用 WeightedRanker 進行混合搜尋
            rerank = WeightedRanker(sparse_weight, dense_weight)
            results = self.collection.hybrid_search(
                [sparse_req, dense_req],
                rerank=rerank,
                limit=limit,
                output_fields=["pk", "text", "data", "content"]
            )
            
            if not results:
                logger.warning("No results found in hybrid search")
                return []
                
            logger.info(f"Hybrid search returned {len(results)} results")
            return results[0] if results else []
            
        except Exception as e:
            logger.error(f"Error in hybrid_search: {str(e)}", exc_info=True)
            return [] 