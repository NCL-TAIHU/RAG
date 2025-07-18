import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model
from typing import List, Optional, Dict, Any, Type
from datetime import datetime
from loguru import logger
from enum import Enum

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.run.adaptor import AppFactory
from src.core.app import App
from src.core.document import Document
from src.core.filter import Filter
from src.utils.metrics import MetricsTracker

# === Enums ===
class SearchMethod(str, Enum):
    DENSE = "dense_search"
    SPARSE = "sparse_search"
    HYBRID = "hybrid_search"

class DatasetType(str, Enum):
    NCL = "ncl"
    LITSEARCH = "litsearch"

# === Base Models ===
class SearchResult(BaseModel):
    """Search result structure"""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    abstract: str = Field(..., description="Document abstract")

class SearchRequest(BaseModel):
    """Search request structure"""
    query: str = Field(..., description="Search query string")
    dataset: DatasetType = Field(DatasetType.NCL, description="Dataset type to search in")
    method: SearchMethod = Field(SearchMethod.HYBRID, description="Search method to use")
    limit: int = Field(5, ge=1, le=100, description="Maximum number of results to return")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter criteria - structure depends on dataset type")

class SearchResponse(BaseModel):
    """Search response structure"""
    results: List[SearchResult] = Field(..., description="List of search results")
    llm_response: str = Field(..., description="LLM-generated response based on results")
    query_time: float = Field(..., description="Query execution time in seconds")
    dataset: str = Field(..., description="Dataset that was searched")
    filter_schema: Dict[str, Any] = Field(..., description="Available filter fields for this dataset")

# === FastAPI App ===
app = FastAPI(
    title="Document-Agnostic RAG Search API",
    description="""
    A document-agnostic RAG (Retrieval-Augmented Generation) API for searching and generating responses.
    
    **Supported Features:**
    - Multiple dataset types (ncl, litsearch)
    - Dynamic filter schemas based on document type
    - Dense, sparse, and hybrid search methods
    - APP pattern architecture with AppFactory and SearchApp
    
    **Filter Structure:**
    The filter structure is automatically generated from the document schema.
    Use the `/schema/{dataset}` endpoint to get available filter fields for each dataset.
    """,
    version="2.0.0"
)

# === Global State ===
# Cache for initialized SearchApp instances per dataset
search_apps: Dict[str, App] = {}
document_classes: Dict[str, Type[Document]] = {}
filter_classes: Dict[str, Type[Filter]] = {}
metrics_tracker = MetricsTracker()

# === Helper Functions ===
async def get_or_create_search_app(dataset: str) -> tuple[App, Type[Document], Type[Filter]]:
    """Get or create SearchApp for the specified dataset using the APP pattern."""
    if dataset not in search_apps:
        logger.info(f"Initializing SearchApp for dataset: {dataset}")
        
        # Create SearchApp using AppFactory pattern (like in main.py)
        factory = AppFactory.from_default(name="dev", dataset=dataset)
        search_app = factory.build()
        
        # Initialize LLM attribute (will be lazily initialized in rag method)
        search_app.llm = None
        
        # Create document and filter classes
        doc_cls = Document.from_dataset(dataset)
        filter_cls = Filter.from_dataset(dataset)
        
        search_apps[dataset] = search_app
        document_classes[dataset] = doc_cls
        filter_classes[dataset] = filter_cls
        
        logger.info(f"✅ SearchApp for dataset {dataset} initialized successfully.")
    
    return search_apps[dataset], document_classes[dataset], filter_classes[dataset]

def generate_filter_schema(filter_cls: Type[Filter]) -> Dict[str, Any]:
    """Generate dynamic filter schema from filter class."""
    filter_fields = filter_cls.filter_fields()
    must_fields = filter_cls.must_fields()
    
    return {
        "filter_fields": filter_fields,
        "must_fields": must_fields
    }

def format_search_results(results: List[Document], doc_cls: Type[Document]) -> List[SearchResult]:
    """Format core Document objects to API SearchResult objects (document-agnostic with Chinese priority)."""
    formatted_results = []
    
    for hit in results:
        # Extract title and content from document content fields
        content_fields = hit.channels()
        
        title = ""
        content = ""
        
        # Priority 1: Look for Chinese title and abstract first (document-agnostic via naming)
        for field_name, field_data in content_fields.items():
            if "title" in field_name.lower() and "chinese" in field_name.lower() and field_data.contents:
                title = field_data.contents[0]
            elif "abstract" in field_name.lower() and "chinese" in field_name.lower() and field_data.contents:
                content = field_data.contents[0]
        
        # Priority 2: Fallback to any title/abstract if Chinese not found
        if not title or not content:
            for field_name, field_data in content_fields.items():
                if not title and "title" in field_name.lower() and field_data.contents:
                    title = field_data.contents[0]
                elif not content and "abstract" in field_name.lower() and field_data.contents:
                    content = field_data.contents[0]
        
        # Priority 3: Final fallback to any available content
        if not title and content_fields:
            first_field = next(iter(content_fields.values()))
            if first_field.contents:
                title = first_field.contents[0]
        
        if not content and content_fields:
            for field_data in content_fields.values():
                if field_data.contents:
                    content = field_data.contents[0]
                    break
        
        formatted_results.append(SearchResult(
            id=hit.key(),
            title=title or "No title available",
            content=content or "No content available", 
            abstract=content or "No abstract available",
        ))
    
    return formatted_results

# === API Endpoints ===
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "Document-Agnostic RAG Search API",
        "version": "2.0.0",
        "status": "operational",
        "supported_datasets": [dataset.value for dataset in DatasetType],
        "description": "A document-agnostic search API using APP pattern with AppFactory and SearchApp"
    }

@app.get("/schema/{dataset}", tags=["Schema"])
async def get_filter_schema(dataset: DatasetType):
    """Get the available filter fields for a specific dataset."""
    try:
        filter_cls = Filter.from_dataset(dataset.value)
        
        return {
            "dataset": dataset.value,
            "filter_fields": filter_cls.filter_fields(),
            "must_fields": filter_cls.must_fields(),
            "description": {
                "filter_fields": "OR logic - documents matching ANY of the specified values",
                "must_fields": "AND logic - documents must contain ALL specified values",
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get schema for dataset {dataset}: {str(e)}")

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Search for documents and generate a response using RAG.
    
    **Parameters:**
    - **query**: The search query string
    - **dataset**: Dataset type (ncl, litsearch) - determines document schema and available filters
    - **method**: Search method (dense_search, sparse_search, or hybrid_search)  
    - **limit**: Maximum number of results to return
    - **filter**: Filter criteria based on dataset schema (use `/schema/{dataset}` to see available fields)
    
    **Filter Usage:**
    The filter structure depends on the dataset type. Each dataset has:
    - **filter_fields**: OR logic - documents matching ANY of the specified values
    - **must_fields**: AND logic - documents must contain ALL specified values
    
    **Example for NCL dataset:**
    ```json
    {
      "filter": {
        "year": [2020, 2021],           // OR: year is 2020 OR 2021  
        "category": ["博士", "碩士"],    // OR: category is 博士 OR 碩士
        "authors_chinese": ["張三"]      // AND: must contain author 張三
      }
    }
    ```
    """
    try:
        metrics_tracker.start_tracking()
        start_time = datetime.now()
        
        # Get or create SearchApp for the dataset using APP pattern
        search_app, doc_cls, filter_cls = await get_or_create_search_app(request.dataset.value)
        
        # Convert API filter to core filter
        core_filter = filter_cls.EMPTY  # Start with empty filter
        if request.filter:
            try:
                # Simple type conversion: convert integer years to strings
                converted_filter = {}
                for field_name, values in request.filter.items():
                    if field_name == "year" and isinstance(values, list):
                        # Convert year integers to strings
                        converted_filter[field_name] = [str(v) for v in values]
                    else:
                        converted_filter[field_name] = values
                
                core_filter = filter_cls(**converted_filter)
            except Exception as e:
                # Get available fields for error message
                available_fields = filter_cls.filter_fields() + filter_cls.must_fields()
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid filter fields. Available fields for {request.dataset.value}: {available_fields}. Error: {str(e)}"
                )
        
        # Perform search using the SearchApp interface (like in main.py)
        results = search_app.search(
            query=request.query,
            filter=core_filter,
            limit=request.limit
        )

        if not results:
            return SearchResponse(
                results=[],
                llm_response="抱歉，找不到相關的結果。",
                query_time=(datetime.now() - start_time).total_seconds(),
                dataset=request.dataset.value,
                filter_schema=generate_filter_schema(filter_cls)
            )

        # Generate LLM response using the SearchApp RAG method (like in main.py)
        rag_result = search_app.rag(request.query, results)
        llm_response = rag_result["generation"]
        llm_prompt = rag_result["prompt"]
        
        # Format results
        formatted_results = format_search_results(results, doc_cls)

        # Track metrics
        metrics = metrics_tracker.end_tracking(
            query=request.query,
            llm_response=llm_response,
            prompt=llm_prompt,
            results=formatted_results
        )
        
        return SearchResponse(
            results=formatted_results,
            llm_response=llm_response,
            query_time=metrics["duration"],
            dataset=request.dataset.value,
            filter_schema=generate_filter_schema(filter_cls)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜尋操作失敗: {str(e)}")

# === Entrypoint ===
if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# === Example Usage ===
"""
# NCL Dataset Example:
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "機器學習在自然語言處理中的應用",
           "dataset": "ncl",
           "method": "hybrid_search",
           "limit": 5,
           "filter": {
             "year": [2020, 2021, 2022],
             "category": ["博士"],
             "keywords": ["Machine Learning", "NLP"],
             "authors_chinese": ["張三"]
           }
         }'

# LitSearch Dataset Example:
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "deep learning for computer vision",
           "dataset": "litsearch", 
           "method": "hybrid_search",
           "limit": 5,
           "filter": {
             "year": ['2020', '2021'],
             "venue": ["ICCV", "CVPR"],
             "authors": ["Yann LeCun"]
           }
         }'

# Get available filter fields for a dataset:
curl -X GET "http://localhost:8000/schema/ncl"
curl -X GET "http://localhost:8000/schema/litsearch"
"""