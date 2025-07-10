# import os
# import sys
# import uvicorn
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from typing import List, Optional
# from datetime import datetime
# from loguru import logger
# from enum import Enum

# # Add the project root directory to Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from src.run.app import SearchApp
# from src.core.data import DataLoader
# from src.core.library import InMemoryLibrary
# from src.core.embedder import BGEM3Embedder, AutoModelEmbedder
# from src.core.search_engine import HybridSearchEngine, MilvusSearchEngine, ElasticSearchEngine, Filter
# from src.core.manager import Manager
# from src.utils.metrics import MetricsTracker

# # === Enums ===
# class SearchMethod(str, Enum):
#     DENSE = "dense_search"
#     SPARSE = "sparse_search"
#     HYBRID = "hybrid_search"

# # === Base Models ===
# class SearchFilter(BaseModel):
#     """Filter criteria for search results"""
#     ids: Optional[List[str]] = Field(None, description="List of document IDs to filter by")
#     years: Optional[List[int]] = Field(None, description="List of years to filter by")
#     categories: Optional[List[str]] = Field(None, description="List of categories to filter by")
#     schools: Optional[List[str]] = Field(None, description="List of schools to filter by")
#     depts: Optional[List[str]] = Field(None, description="List of departments to filter by")
#     keywords: List[str] = Field(default_factory=list, description="List of keywords to filter by")
#     authors: List[str] = Field(default_factory=list, description="List of authors to filter by")
#     advisors: List[str] = Field(default_factory=list, description="List of advisors to filter by")

# class SearchResult(BaseModel):
#     """Search result structure"""
#     id: str = Field(..., description="Document ID")
#     title: str = Field(..., description="Document title")
#     content: str = Field(..., description="Document content")
#     abstract: str = Field(..., description="Document abstract")
#     score: float = Field(..., description="Search relevance score")

# class SearchRequest(BaseModel):
#     """Search request structure"""
#     query: str = Field(..., description="Search query string")
#     method: SearchMethod = Field(SearchMethod.HYBRID, description="Search method to use")
#     limit: int = Field(5, ge=1, le=100, description="Maximum number of results to return")
#     filter: Optional[SearchFilter] = Field(None, description="Filter criteria for search results")

# class SearchResponse(BaseModel):
#     """Search response structure"""
#     results: List[SearchResult] = Field(..., description="List of search results")
#     llm_response: str = Field(..., description="LLM-generated response based on results")
#     query_time: float = Field(..., description="Query execution time in seconds")

# # === FastAPI App ===
# app = FastAPI(
#     title="RAG Search API",
#     description="""
#     A simple RAG (Retrieval-Augmented Generation) API for searching and generating responses.
#     Supports dense search, sparse search, and hybrid search methods.
#     Uses Elasticsearch for filtering and Milvus for vector search.
#     """,
#     version="1.0.0"
# )

# # === Component Initialization ===
# def initialize_components():
#     """Initialize all required components for the search application"""
#     try:
#         dataloader = DataLoader.from_default("ncl")
#         library = InMemoryLibrary()
#         sparse_embedder = BGEM3Embedder(model_name="BAAI/bge-m3")
#         dense_embedder = AutoModelEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
#         # 初始化 Elasticsearch 和 Milvus 搜尋引擎
#         elastic_engine = ElasticSearchEngine(
#             es_host="https://localhost:9200",
#             es_index="documents",
#         )
#         milvus_engine = MilvusSearchEngine(sparse_embedder, dense_embedder)
        
#         # 創建混合搜尋引擎
#         hybrid_engine = HybridSearchEngine(
#             relational_search_engine=elastic_engine,
#             vector_search_engine=milvus_engine
#         )
        
#         manager = Manager(library, [hybrid_engine], router_name="simple")
#         search_app = SearchApp(dataloader, manager, max_files=float('inf'))
        
#         return search_app
#     except Exception as e:
#         logger.error(f"Failed to initialize components: {e}")
#         raise

# search_app = initialize_components()
# metrics_tracker = MetricsTracker()

# # === API Endpoints ===
# @app.on_event("startup")
# async def startup_event():
#     """Initialize the search engine on startup"""
#     try:
#         logger.info("🔧 Checking database status...")
        
#         # 檢查 Elasticsearch 索引是否存在
#         es_engine = search_app.manager.search_engines[0].relational_search_engine
#         milvus_engine = search_app.manager.search_engines[0].vector_search_engine
        
#         es_exists = es_engine.es.indices.exists(index=es_engine.es_index)
#         milvus_exists = hasattr(milvus_engine, 'operator') and milvus_engine.operator is not None
        
#         if es_exists and milvus_exists:
#             logger.info(f"✅ Elasticsearch index {es_engine.es_index} and Milvus collection already exist.")
#         else:
#             logger.info("🔄 Initializing search engine...")
#             search_app.setup()
#             logger.info("✅ Search engine initialized successfully.")
            
#     except Exception as e:
#         logger.error(f"❌ Initialization failed: {e}")
#         raise

# @app.get("/", tags=["Health"])
# async def root():
#     """Root endpoint returning API information"""
#     return {
#         "name": "RAG Search API",
#         "version": "1.0.0",
#         "status": "operational"
#     }

# @app.post("/search", response_model=SearchResponse, tags=["Search"])
# async def search(request: SearchRequest):
#     """
#     Search for documents and generate a response using RAG.
    
#     - **query**: The search query string
#     - **method**: Search method (dense_search, sparse_search, or hybrid_search)
#     - **limit**: Maximum number of results to return
#     - **filter**: Optional filter criteria for Elasticsearch filtering
#     """
#     try:
#         # 開始追蹤指標
#         metrics_tracker.start_tracking()
        
#         start_time = datetime.now()
        
#         # 轉換 filter 為 Elasticsearch 可用的格式
#         es_filter = Filter()  # 創建一個空的 Filter 物件
#         if request.filter:
#             es_filter = Filter(
#                 ids=request.filter.ids,
#                 years=request.filter.years,
#                 categories=request.filter.categories,
#                 schools=request.filter.schools,
#                 depts=request.filter.depts,
#                 keywords=request.filter.keywords,
#                 authors=request.filter.authors,
#                 advisors=request.filter.advisors
#             )
        
#         # Perform search
#         results = search_app.search(
#             query=request.query,
#             filter=es_filter,
#             limit=request.limit
#         )

#         if not results:
#             return SearchResponse(
#                 results=[],
#                 llm_response="抱歉，找不到相關的結果。",
#                 query_time=(datetime.now() - start_time).total_seconds()
#             )

#         # Generate LLM response
#         rag_result = search_app.rag(request.query, results)
#         llm_response = rag_result["generation"]
#         llm_prompt = rag_result["prompt"]
        
#         print("--------------------------------")
#         print(f"Prompt: {llm_prompt}")
#         print("--------------------------------")
#         print(f"Raw results: {llm_response}")
#         print("--------------------------------")
        
#         # Format results
#         formatted_results = []
#         for hit in results:
#             # 從 chinese 或 english 中提取 title
#             title = hit.chinese.title if hit.chinese and hit.chinese.title else \
#                     hit.english.title if hit.english and hit.english.title else "無標題"
            
#             # 從 chinese 或 english 中提取 abstract 作為 content
#             content = hit.chinese.abstract if hit.chinese and hit.chinese.abstract else \
#                       hit.english.abstract if hit.english and hit.english.abstract else ""
            
#             formatted_results.append(SearchResult(
#                 id=hit.id,
#                 title=title,
#                 content=content,
#                 abstract=content,  # 這裡用 content 作為 abstract
#                 score=hit.score if hasattr(hit, "score") else 0.0
#             ))

#         print("--------------------------------")
#         print(f"Formatted results: {formatted_results}")
#         print("--------------------------------")


#         # 結束追蹤並記錄指標
#         metrics = metrics_tracker.end_tracking(
#             query=request.query,
#             llm_response=llm_response,
#             prompt=llm_prompt,
#             results=formatted_results
#         )
        
#         return SearchResponse(
#             results=formatted_results,
#             llm_response=llm_response,
#             query_time=metrics["duration"]
#         )

#     except Exception as e:
#         logger.error(f"Search error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"搜尋操作失敗: {str(e)}")

# # === Entrypoint ===
# if __name__ == "__main__":
#     port = int(os.getenv("API_PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port) 



# # Full Example
# '''
# curl -X POST "http://localhost:8000/search" \
#      -H "Content-Type: application/json" \
#      -d '{
#            "query": "機器學習在自然語言處理中的應用",
#            "method": "hybrid_search",
#            "limit": 5,
#            "filter": {
#              "years": [2020, 2021, 2022],
#              "categories": ["Computer Science"],
#              "keywords": ["Machine Learning", "NLP"]
#            }
#          }'
# '''