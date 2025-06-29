import os
import argparse
from loguru import logger
from src.utils.config import Config
from src.core.data_processor import DataProcessor
from src.core.database import DatabaseManager
from src.core.search import SearchEngine
from src.core.llm_manager import LLMManager
import logging
from typing import List, Optional, Any
from datetime import datetime

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'hybrid_search_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HybridSearchApp:
    """混合搜尋應用程式主類別"""
    
    def __init__(self):
        """初始化應用程式"""
        self.data_processor = DataProcessor()
        self.db_manager = DatabaseManager()
        self.search_engine = None
        self.llm_manager = None
    
    def setup(self):
        """設定應用程式"""
        try:
            # 初始化嵌入函數
            ef = self.data_processor.initialize_embedding_function()
            
            if not os.path.exists(Config.DB_URI):
                # 處理資料
                metadata, abstract = self.data_processor.load_and_process_data()
                
                # 生成嵌入
                docs_embeddings, dense_dim = self.data_processor.generate_embeddings(abstract, ef)
                
                # 設定資料庫
                collection = self.db_manager.setup_database(metadata, abstract, docs_embeddings, dense_dim)
                logger.info(f"Database setup completed with {collection.num_entities} entities")
            else:
                logger.info("Using existing database")
                collection = self.db_manager.get_collection()
            
            # 初始化搜尋引擎
            self.search_engine = SearchEngine(collection)
            
            # 初始化LLM管理器
            self.llm_manager = LLMManager(Config.LLM_MODEL)
            
        except Exception as e:
            logger.error(f"Error in setup: {str(e)}")
            raise
    
    def generate_prompt(self, retrieved_doc: list, query: str) -> str:
        """生成提示詞"""
        try:
            prompt = ""
            for i, doc in enumerate(retrieved_doc):
                prompt = prompt + f"{i+1}.\n{doc.get('data')}\n{doc.get('text')}\n"
                            
            prompt += (
                f"根據以上資料，請針對以下問題進行整合與分析：\n"
                f"問題：{query}\n\n"
                "請依照以下指引生成回答：\n"
                "- 用一段文字統整這些論文可能涉及的核心研究主題與趨勢，避免逐一重複每篇內容。\n"
                "- 請在這段文字中自然地帶出 2 到 3 個可能的後續研究方向。\n\n"
                "請注意：\n"
                "- 不要說『本文』或『以上回應』，直接給出分析結果。\n"
                "- 不要使用條列格式、標題、步驟編號、格式化開頭（如『本文將…』）。\n"
                "- 文字需自然流暢、邏輯清晰，不要重複句式或重複前面資料內容。\n"
            )
        

            return prompt
        except Exception as e:
            logger.error(f"Error in generate_prompt: {str(e)}")
            raise
    
    def search(self, query: str, method: str = "hybrid_search", 
              sparse_weight: float = None, dense_weight: float = None,
              limit: int = None) -> list:
        """執行搜尋
        
        Args:
            query: 搜尋查詢
            method: 搜尋方法 (dense_search/sparse_search/hybrid_search)
            sparse_weight: 稀疏向量權重
            dense_weight: 密集向量權重
            limit: 結果數量限制
            
        Returns:
            list: 搜尋結果
        """
        try:
            logger.info(f"Starting search with method: {method}, query: {query}")
            
            # 生成查詢嵌入
            logger.info("Initializing embedding function...")
            ef = self.data_processor.initialize_embedding_function()
            logger.info("Generating query embeddings...")
            query_embeddings = ef([query])
            
            if not query_embeddings:
                logger.error("Failed to generate query embeddings")
                return []
                
            logger.info("Query embeddings generated successfully")
            
            # 執行搜尋
            if method == "dense_search":
                logger.info("Performing dense search...")
                if "dense" not in query_embeddings:
                    logger.error("Dense embeddings not found in query_embeddings")
                    return []
                results = self.search_engine.dense_search(query_embeddings["dense"][0], limit)
            elif method == "sparse_search":
                logger.info("Performing sparse search...")
                if "sparse" not in query_embeddings:
                    logger.error("Sparse embeddings not found in query_embeddings")
                    return []
                results = self.search_engine.sparse_search(query_embeddings["sparse"]._getrow(0), limit)
            elif method == "hybrid_search":
                logger.info("Performing hybrid search...")
                if "dense" not in query_embeddings or "sparse" not in query_embeddings:
                    logger.error("Missing required embeddings for hybrid search")
                    return []
                results = self.search_engine.hybrid_search(
                    query_embeddings["dense"][0],
                    query_embeddings["sparse"]._getrow(0),
                    sparse_weight=sparse_weight,
                    dense_weight=dense_weight,
                    limit=limit
                )
            else:
                logger.error(f"Invalid search method: {method}")
                raise ValueError(f"Invalid search method: {method}")
            
            logger.info(f"Search returned {len(results)} results")
            
            # 確保結果是列表
            if not isinstance(results, list):
                logger.error(f"Search results is not a list: {type(results)}")
                return []
                
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}", exc_info=True)
            return []
    
    def generate_response(self, results: list, query: str) -> str:
        """生成回應"""
        try:
            prompt = self.generate_prompt(results, query)
            logger.info(f"Generating response using model: {Config.LLM_MODEL}")
            
            response = self.llm_manager.generate(prompt)
            
            logger.info("Generated response:")
            logger.info("="*50)
            logger.info(response)
            logger.info("="*50)
            
            return response
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            raise

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="Hybrid Search Application")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--method", choices=["dense_search", "sparse_search", "hybrid_search"],
                      default="hybrid_search", help="Search method")
    parser.add_argument("--sparse-weight", type=float, default=0.5, help="Sparse weight for hybrid search")
    parser.add_argument("--dense-weight", type=float, default=0.5, help="Dense weight for hybrid search")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    try:
        app = HybridSearchApp()
        app.setup()
        
        # 執行搜尋
        results = app.search(
            query=args.query,
            method=args.method,
            sparse_weight=args.sparse_weight,
            dense_weight=args.dense_weight,
            limit=args.limit
        )
        
        # 顯示搜尋結果
        print("\nSearch Results:")
        for i, hit in enumerate(results, 1):
            try:
                # 正確處理 Milvus 的 Hit 物件
                data = hit.entity.get('data')
                logger.debug(f"Document {i} data: {data}")
                
                if not data:
                    logger.warning(f"Document {i} has no data field")
                    print(f"{i}. [No data available]")
                    continue
                
                # 安全地獲取第一行
                lines = data.splitlines()
                if not lines:
                    logger.warning(f"Document {i} data is empty")
                    print(f"{i}. [Empty document]")
                    continue
                
                title = lines[0]
                print(f"{i}. {title}")
                
            except Exception as e:
                logger.error(f"Error processing document {i}: {str(e)}")
                print(f"{i}. [Error processing document]")
        
        # 生成回應
        response = app.generate_response(results, args.query)
        print("\nLLM Response:")
        print(response)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 