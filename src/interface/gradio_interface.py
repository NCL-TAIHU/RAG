import gradio as gr
import requests
import json
from typing import List, Dict, Any
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000"

def search_documents(
    query: str,
    method: str,
    limit: int,
    years: List[int],
    categories: List[str],
    # schools: List[str],
    # depts: List[str],
    keywords: List[str],
    # authors: List[str],
    # advisors: List[str]
) -> Dict[str, Any]:
    """
    執行文檔搜尋並返回結果
    """
    # 構建搜尋請求
    search_request = {
        "query": query,
        "method": method,
        "limit": limit,
        "filter": {
            "years": years if years else None,
            "categories": categories if categories else None,
            # "schools": schools if schools else None,
            # "depts": depts if depts else None,
            "keywords": keywords if keywords else [],
            # "authors": authors if authors else [],
            # "advisors": advisors if advisors else []
        }
    }
    
    try:
        # 發送請求到 API
        response = requests.post(
            f"{API_URL}/search",
            json=search_request,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    格式化搜尋結果為易讀的文本
    """
    if not results:
        return "沒有找到相關結果。"
    
    formatted_text = "📚 相關文獻：\n\n"
    for i, result in enumerate(results, 1):
        formatted_text += f"{i}. {result['title']}\n"
    
    return formatted_text

def create_interface():
    """
    創建 Gradio 介面
    """
    with gr.Blocks(title="RAG 智能助手", theme=gr.themes.Soft()) as interface:
        # 聊天歷史
        chatbot = gr.Chatbot(
            label="對話歷史",
            height=600,
            show_copy_button=True,
            avatar_images=("👤", "🤖")
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                # 搜尋輸入
                query = gr.Textbox(
                    label="請輸入您的問題",
                    placeholder="例如：請告訴我機器學習在自然語言處理中的應用...",
                    lines=2,
                    show_label=False
                )
            
            with gr.Column(scale=1):
                # 搜尋按鈕
                search_btn = gr.Button("發送", variant="primary", size="lg")
        
        with gr.Accordion("進階搜尋設定", open=False):
            with gr.Row():
                with gr.Column():
                    # 篩選條件
                    years = gr.Textbox(
                        label="年份 (用逗號分隔)",
                        placeholder="例如：108,109,110"
                    )
                    categories = gr.Textbox(
                        label="類別 (用逗號分隔)",
                        placeholder="例如：碩士,博士"
                    )
                    # schools = gr.Textbox(
                    #     label="學校 (用逗號分隔)",
                    #     placeholder="例如：國立清華大學"
                    # )
                    # depts = gr.Textbox(
                    #     label="系所 (用逗號分隔)",
                    #     placeholder="例如：資訊工程學系"
                    # )
                    keywords = gr.Textbox(
                        label="關鍵字 (用逗號分隔)",
                        placeholder="例如：機器學習,深度學習"
                    )
                    # authors = gr.Textbox(
                    #     label="作者 (用逗號分隔)",
                    #     placeholder="例如：張三,李四"
                    # )
                    # advisors = gr.Textbox(
                    #     label="指導教授 (用逗號分隔)",
                    #     placeholder="例如：王五,趙六"
                    # )
        
        def process_search(
            query: str,
            years: str,
            categories: str,
            # schools: str,
            # depts: str,
            keywords: str,
            # authors: str,
            # advisors: str,
            history: List[List[str]]
        ) -> tuple:
            if not query.strip():
                return history
            
            # 處理輸入的篩選條件
            years_list = [int(y.strip()) for y in years.split(",")] if years else []
            categories_list = [c.strip() for c in categories.split(",")] if categories else []
            # schools_list = [s.strip() for s in schools.split(",")] if schools else []
            # depts_list = [d.strip() for d in depts.split(",")] if depts else []
            keywords_list = [k.strip() for k in keywords.split(",")] if keywords else []
            # authors_list = [a.strip() for a in authors.split(",")] if authors else []
            # advisors_list = [adv.strip() for adv in advisors.split(",")] if advisors else []
            
            # 執行搜尋（使用固定的搜尋方法和結果數量）
            response = search_documents(
                query=query,
                method="hybrid_search",  # 固定使用混合搜尋
                limit=5,  # 固定顯示5個結果
                years=years_list,
                categories=categories_list,
                # schools=schools_list,
                # depts=depts_list,
                keywords=keywords_list,
                # authors=authors_list,
                # advisors=advisors_list
            )
            
            if "error" in response:
                history.append([query, f"❌ 發生錯誤：{response['error']}"])
                return history
            
            # 格式化結果
            results_text = format_search_results(response["results"])
            llm_text = response["llm_response"]
            
            # 組合回應
            full_response = f"{llm_text}\n\n{results_text}\n\n⏱️ 查詢時間：{response['query_time']:.2f} 秒"
            
            # 更新歷史
            history.append([query, full_response])
            return history
        
        # 設置事件處理
        search_btn.click(
            fn=process_search,
            inputs=[
                query,
                years, categories, # schools, depts,
                keywords, # authors, advisors,
                chatbot
            ],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",  # 清空輸入框
            inputs=[],
            outputs=[query]
        )
        
        # 按 Enter 發送
        query.submit(
            fn=process_search,
            inputs=[
                query,
                years, categories, # schools, depts,
                keywords, # authors, advisors,
                chatbot
            ],
            outputs=[chatbot]
        ).then(
            fn=lambda: "",  # 清空輸入框
            inputs=[],
            outputs=[query]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 