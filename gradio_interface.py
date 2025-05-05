import gradio as gr
import requests
import os
from loguru import logger
import json
from typing import Dict, List, Tuple, Any
import pandas as pd

# 配置
API_URL = os.getenv("API_URL", "http://localhost:8000")

def format_search_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """格式化搜尋結果為 pandas DataFrame"""
    if not results:
        return pd.DataFrame(columns=["標題"])
    
    titles = []
    for result in results:
        title = result.get("title", "無標題")
        titles.append(title)
    
    return pd.DataFrame({"標題": titles})

def perform_search(
    query: str,
    method: str = "hybrid_search",
    sparse_weight: float = 0.5,
    dense_weight: float = 0.5,
    limit: int = 5
) -> Tuple[pd.DataFrame, str]:
    """執行搜尋，並回傳結果及 LLM 回應"""
    try:
        # 準備請求
        payload = {
            "query": query,
            "method": method,
            "sparse_weight": sparse_weight,
            "dense_weight": dense_weight,
            "limit": limit
        }
        
        # 發送請求
        response = requests.post(f"{API_URL}/search", json=payload)
        
        # 檢查回應
        if response.status_code != 200:
            error_msg = f"API 請求失敗，狀態碼: {response.status_code}"
            logger.error(error_msg)
            return pd.DataFrame(columns=["標題"]), error_msg
        
        # 解析回應
        data = response.json()
        results = data.get("results", [])
        llm_response = data.get("llm_response", "沒有找到相關回應")
        
        # 格式化結果
        df = format_search_results(results)
        
        return df, llm_response
        
    except Exception as e:
        error_msg = f"搜尋過程中發生錯誤: {str(e)}"
        logger.error(error_msg)
        return pd.DataFrame(columns=["標題"]), error_msg

def create_interface():
    """創建 Gradio 界面"""
    with gr.Blocks(title="論文混合搜尋系統", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 論文混合搜尋系統")
        gr.Markdown("使用混合向量搜尋方法在論文資料庫中尋找相關資料，並由 AI 進行總結")
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="搜尋問題",
                    placeholder="例如: 人工智能在醫療中的應用...",
                    lines=2
                )
            
            with gr.Column(scale=1):
                search_btn = gr.Button("搜尋", variant="primary")
        
        with gr.Accordion("進階設定", open=False):
            with gr.Row():
                method = gr.Radio(
                    ["hybrid_search", "dense_search", "sparse_search"],
                    label="搜尋方法",
                    value="hybrid_search"
                )
                
            with gr.Row():
                sparse_weight = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="稀疏向量權重"
                )
                
                dense_weight = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="密集向量權重"
                )
                
                limit = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="顯示結果數量"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 搜尋結果")
                results_df = gr.Dataframe(headers=["標題"], interactive=False)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## AI 總結")
                llm_response = gr.Textbox(label="摘要", lines=6, interactive=False)
        
        search_btn.click(
            fn=perform_search,
            inputs=[query_input, method, sparse_weight, dense_weight, limit],
            outputs=[results_df, llm_response]
        )
        
        gr.Markdown("---")
        gr.Markdown("### 說明")
        gr.Markdown("""
        * **混合搜尋 (hybrid_search)**: 同時使用稀疏向量和密集向量進行搜尋
        * **密集向量搜尋 (dense_search)**: 使用語義相似度進行搜尋
        * **稀疏向量搜尋 (sparse_search)**: 使用關鍵字相似度進行搜尋
        * **權重**: 調整稀疏向量與密集向量在混合搜尋中的重要性
        """)
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    port = int(os.getenv("GRADIO_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False) 