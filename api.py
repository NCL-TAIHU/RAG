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
        
        if not results:
            return SearchResponse(results=[], llm_response="No results found")
        
        # 生成 LLM 回應
        llm_response = search_app.generate_response(results, request.query)
        
        # 格式化結果
        formatted_results = []
        for hit in results:
            try:
                # 改用屬性訪問方式，避免 Hit.get() 的問題
                data = getattr(hit.entity, 'data', '')
                content = getattr(hit.entity, 'content', '')
                
                # 擷取標題 (第一行)
                lines = data.splitlines() if data else []
                title = lines[0] if lines else "No title available"
                
                formatted_results.append(SearchResult(
                    title=title,
                    data=data,
                    content=content
                ))
            except Exception as e:
                logger.error(f"Error formatting result: {str(e)}")
        
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