import os
import uvicorn
import threading
import gradio as gr
from src.api.api import app
from src.interface.gradio_interface import create_interface
from loguru import logger
import time

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
        demo.launch(server_name="0.0.0.0", server_port=port, share=True)
    except Exception as e:
        logger.error(f"Error starting Gradio interface: {str(e)}")
        raise

def main():
    """主函數：同時啟動 API 和 Gradio"""
    try:
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
        raise

if __name__ == "__main__":
    main() 