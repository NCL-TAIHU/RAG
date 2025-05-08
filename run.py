import os
import uvicorn
import threading
import gradio as gr
from src.api.api import app
from src.interface.gradio_interface import create_interface
from loguru import logger
import time
from src.core.data_processor import DataProcessor
from src.core.database import DatabaseManager
from src.utils.config import Config
import sys

# 配置日誌
logger.remove()  # 移除預設的處理器
logger.add(sys.stderr, level="INFO")  # 添加控制台輸出
logger.add("logs/run_{time}.log", rotation="500 MB")  # 添加文件輸出

def initialize_database():
    """初始化 Milvus 數據庫"""
    try:
        logger.info("Checking database initialization...")
        logger.info(f"Database path: {Config.DB_URI}")
        logger.info(f"Abstract path: {Config.ABSTRACT_PATH}")
        logger.info(f"Data path: {Config.DATA_PATH}")
        logger.info(f"Content path: {Config.CONTENT_PATH}")
        
        if not os.path.exists(Config.DB_URI):
            logger.info("Database not found. Starting initialization...")
            
            # 檢查必要的路徑是否存在
            for path in [Config.ABSTRACT_PATH, Config.DATA_PATH, Config.CONTENT_PATH]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Required path does not exist: {path}")
            
            # 初始化數據處理器
            logger.info("Initializing data processor...")
            data_processor = DataProcessor()
            
            # 初始化數據庫管理器
            logger.info("Initializing database manager...")
            db_manager = DatabaseManager()
            
            # 初始化嵌入函數
            logger.info("Initializing embedding function...")
            ef = data_processor.initialize_embedding_function()
            logger.info("Embedding function initialized successfully")
            
            # 處理數據
            logger.info("Loading and processing data...")
            metadata, abstract = data_processor.load_and_process_data()
            logger.info(f"Processed {len(abstract)} documents")
            
            # 生成嵌入
            logger.info("Generating embeddings...")
            docs_embeddings, dense_dim = data_processor.generate_embeddings(abstract, ef)
            logger.info(f"Generated embeddings with dimension {dense_dim}")
            
            # 設定數據庫
            logger.info("Setting up database...")
            collection = db_manager.setup_database(metadata, abstract, docs_embeddings, dense_dim)
            logger.info(f"Database initialization completed with {collection.num_entities} entities")
        else:
            logger.info("Database already exists. Skipping initialization.")
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during database initialization: {str(e)}")
        logger.exception("Detailed error information:")
        raise

def run_api():
    """運行 FastAPI 服務"""
    try:
        port = int(os.getenv("API_PORT", 8000))
        logger.info(f"Starting API server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}")
        raise

def run_gradio():
    """運行 Gradio 界面"""
    try:
        # 等待 API 服務啟動
        time.sleep(2)
        port = int(os.getenv("GRADIO_PORT", 7860))
        logger.info(f"Starting Gradio interface on port {port}")
        demo = create_interface()
        demo.launch(server_name="0.0.0.0", server_port=port, share=False)
    except Exception as e:
        logger.error(f"Error starting Gradio interface: {str(e)}")
        raise

def main():
    """主函數：初始化數據庫並啟動服務"""
    try:
        # 初始化數據庫
        initialize_database()
        
        # 創建並啟動 API 服務線程
        api_thread = threading.Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()

        # 啟動 Gradio 界面
        run_gradio()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.exception("Detailed error information:")
        raise

if __name__ == "__main__":
    main() 