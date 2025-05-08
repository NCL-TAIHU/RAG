import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger
import uvicorn

from main import HybridSearchApp

# 初始化 FastAPI 應用程式
app = FastAPI(
    title="Hybrid Search API",
    description="API for searching academic papers using hybrid search",
    version="1.0.0"
)

# 初始化混合搜尋應用程式
search_app = HybridSearchApp()

# 啟動時進行設定
@app.on_event("startup")
async def startup_event():
    """應用程式啟動時執行的事件"""
    try:
        logger.info("Initializing search application...")
        search_app.setup()
        logger.info("Search application initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing search application: {str(e)}")
        raise

# 定義請求模型
class SearchRequest(BaseModel):
    query: str
    method: str = "hybrid_search"
    sparse_weight: Optional[float] = 0.5
    dense_weight: Optional[float] = 0.5
    limit: Optional[int] = 5

# 定義回應模型
class SearchResult(BaseModel):
    title: str
    data: str
    content: str

class SearchResponse(BaseModel):
    results: List[SearchResult]
    llm_response: str

@app.get("/")
async def root():
    """API 根路徑"""
    return {"message": "Welcome to Hybrid Search API"}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """執行搜尋"""
    try:
        # 檢查搜尋方法是否有效
        valid_methods = ["dense_search", "sparse_search", "hybrid_search"]
        if request.method not in valid_methods:
            raise HTTPException(status_code=400, detail=f"Invalid search method. Must be one of: {valid_methods}")

        # 執行搜尋
        results = search_app.search(
            query=request.query,
            method=request.method,
            sparse_weight=request.sparse_weight,
            dense_weight=request.dense_weight,
            limit=request.limit
        )

        print(f"[DEBUG] type of results: {type(results)}, results = {results}")

        if not results:
            return SearchResponse(results=[], llm_response="No results found")

        # 生成 LLM 回應
        llm_response = search_app.generate_response(results, request.query)

        # 格式化結果
        formatted_results = []
        for hit in results:
            try:
                print(f"[DEBUG] hit type: {type(hit)}, dir(hit): {dir(hit)}")
                
                # 從Hit物件中提取數據
                hit_data = getattr(hit, 'data', {})
                print(f"[DEBUG] hit_data type: {type(hit_data)}, hit_data keys: {hit_data.keys() if isinstance(hit_data, dict) else 'Not a dict'}")
                
                # 嘗試獲取entity
                entity = None
                if isinstance(hit_data, dict) and 'entity' in hit_data:
                    entity = hit_data['entity']
                elif hasattr(hit, 'entity'):
                    entity = hit.entity
                
                print(f"[DEBUG] entity: {entity}")
                
                # 提取標題和內容
                title = "No title available"
                entity_data = ""
                entity_content = ""
                
                # 從entity中提取數據
                if entity is not None:
                    if isinstance(entity, dict):
                        entity_data = entity.get('data', '')
                        entity_content = entity.get('content', '')
                    else:
                        entity_data = getattr(entity, 'data', '')
                        entity_content = getattr(entity, 'content', '')
                
                # 如果還是沒找到，可能需要檢查hit.entity的其他屬性
                if not entity_data and hasattr(hit, 'entity'):
                    print(f"[DEBUG] Checking hit.entity: {dir(hit.entity) if hasattr(hit, 'entity') else 'No entity attribute'}")
                
                print(f"[DEBUG] entity_data: {entity_data}")
                print(f"[DEBUG] entity_content length: {len(str(entity_content)) if entity_content else 0}")
                
                # 確保數據是字串類型
                if not isinstance(entity_data, str):
                    entity_data = str(entity_data) if entity_data is not None else ""
                
                if not isinstance(entity_content, str):
                    entity_content = str(entity_content) if entity_content is not None else ""
                
                # 嘗試提取標題
                if entity_data:
                    lines = entity_data.splitlines()
                    if lines and '論文名稱:' in lines[0]:
                        title = lines[0].strip()
                        title = title.replace("論文名稱: ", "").strip()
                    elif len(lines) > 0:
                        title = lines[0].strip()
                        title = title.replace("論文名稱: ", "").strip()
                
                # 確保標題是字串
                if not isinstance(title, str):
                    title = str(title) if title is not None else "No title available"
                    # 去除title中的"論文名稱:"
                    title = title.replace("論文名稱: ", "").strip()
                
                print(f"[DEBUG] Final title: {title}")
                
                # 創建SearchResult物件
                search_result = SearchResult(
                    title=title,
                    data=entity_data,
                    content=entity_content
                )
                formatted_results.append(search_result)
                
            except Exception as e:
                logger.error(f"Error formatting result: {str(e)}")
                print(f"[DEBUG] Error: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")

        return SearchResponse(
            results=formatted_results,
            llm_response=llm_response
        )

    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)