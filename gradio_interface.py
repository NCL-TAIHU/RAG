import gradio as gr
import requests
import os
from loguru import logger
import pandas as pd
from typing import Dict, List, Tuple, Any

API_URL = os.getenv("API_URL", "http://localhost:8000")

def format_search_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "⚠️ 沒有找到相關論文資料"
    table_md = "| # | 論文標題 |\n|---|-----------|\n"
    for i, item in enumerate(results, 1):
        table_md += f"| {i} | {item.get('title', '無標題')} |\n"
    return table_md

def perform_search(query: str, method: str, sparse_weight: float, dense_weight: float, limit: int) -> Tuple[str, str]:
    try:
        payload = {
            "query": query,
            "method": method,
            "sparse_weight": sparse_weight,
            "dense_weight": dense_weight,
            "limit": limit
        }
        response = requests.post(f"{API_URL}/search", json=payload)
        if response.status_code != 200:
            return "", f"❌ API 請求失敗（狀態碼 {response.status_code}）"
        data = response.json()
        results_markdown = format_search_results(data.get("results", []))
        llm_response = data.get("llm_response", "（無回應）")
        full_response = f"📚 **系統回應：**\n{llm_response}\n\n🔎 **搜尋結果：**\n{results_markdown}"
        return query, full_response
    except Exception as e:
        return "", f"❌ 發生錯誤：{str(e)}"

def create_interface():
    with gr.Blocks(title="台灣國圖論文搜尋 GPT 介面", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🎓 國圖論文智慧搜尋（類 GPT 介面）")

        chatbot = gr.Chatbot(label="對話歷史", height=500, show_label=False)
        chat_history = gr.State([])  # 儲存歷史

        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(placeholder="請輸入您的問題...", lines=2, label=None)
                send_btn = gr.Button("🚀 送出", variant="primary")

        with gr.Accordion("⚙️ 進階設定", open=False):
            method = gr.Radio(["hybrid_search", "dense_search", "sparse_search"],
                              label="搜尋方法", value="hybrid_search")
            sparse_weight = gr.Slider(0.0, 1.0, 0.5, step=0.1, label="稀疏向量權重")
            dense_weight = gr.Slider(0.0, 1.0, 0.5, step=0.1, label="密集向量權重")
            limit = gr.Slider(1, 10, 5, step=1, label="顯示結果數量")

        def chat_and_search(query, method, sparse_weight, dense_weight, limit, history):
            user_query, system_reply = perform_search(query, method, sparse_weight, dense_weight, limit)
            if user_query:
                history.append([user_query, system_reply])
            return history, history

        send_btn.click(
            fn=chat_and_search,
            inputs=[query_input, method, sparse_weight, dense_weight, limit, chat_history],
            outputs=[chatbot, chat_history]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    port = int(os.getenv("GRADIO_PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)
