#!/usr/bin/env python3
"""
Simple example to test the RAG Search API
"""

import requests
import json

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"📊 Supported datasets: {data['supported_datasets']}")
            return True
        else:
            print(f"❌ API not responding: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def get_filter_schema(dataset):
    """Get available filter fields for a dataset."""
    try:
        response = requests.get(f"{API_BASE_URL}/schema/{dataset}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to get schema: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Schema error: {e}")
        return None

def search_documents(query, dataset="ncl", limit=5):
    """Simple search function."""
    payload = {
        "query": query,
        "dataset": dataset,
        "method": "hybrid_search",
        "limit": limit,
        "filter": {}
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/search", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Search failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Search error: {e}")
        return None

def print_results(results):
    """Print search results with comprehensive document details."""
    if not results:
        return
    
    print(f"\n🔍 Dataset: {results['dataset']}")
    print(f"📊 Found {len(results['results'])} results")
    print(f"⏱️  Query time: {results['query_time']:.3f}s")
    print("=" * 80)
    
    print("\n📝 Results:")
    for i, result in enumerate(results['results'], 1):
        print(f"[{i}] {result['title']}")
        if result.get('title_english'):
            print(f"    English Title: {result['title_english']}")
        
        print(f"    Abstract: {result['abstract'][:200]}...")
        if result.get('abstract_english'):
            print(f"    English Abstract: {result['abstract_english'][:200]}...")
        
        if result.get('authors'):
            print(f"    Authors: {', '.join(result['authors'])}")
        if result.get('advisors'):
            print(f"    Advisors: {', '.join(result['advisors'])}")
        
        if result.get('school') or result.get('department'):
            school_dept = []
            if result.get('school'):
                school_dept.append(result['school'])
            if result.get('department'):
                school_dept.append(result['department'])
            print(f"    Institution: {' - '.join(school_dept)}")
        
        if result.get('year') or result.get('category'):
            year_cat = []
            if result.get('year'):
                year_cat.append(str(result['year']))
            if result.get('category'):
                year_cat.append(result['category'])
            print(f"    Year/Category: {' - '.join(year_cat)}")
        
        if result.get('keywords'):
            print(f"    Keywords: {', '.join(result['keywords'][:5])}...")
        
        if result.get('link'):
            print(f"    Link: {result['link']}")
        
        if result.get('llm_questions'):
            print(f"    🤖 LLM Questions:")
            for i, question in enumerate(result['llm_questions'][:3], 1):
                print(f"      {i}. {question}")
            if len(result['llm_questions']) > 3:
                print(f"      ... and {len(result['llm_questions']) - 3} more questions")
        
        print("-" * 80)
    
    print("=" * 80)
    print(f"\n🤖 LLM Response:\n{results['llm_response']}")
    print("=" * 80)

def main():
    """Main function to test the API."""
    print("🚀 RAG Search API Test")
    print("=" * 60)
    
    # Check API health
    if not test_api_health():
        print("Make sure the API server is running on http://localhost:8000")
        return
    
    # Get and show available fields
    print("\n📋 Getting available filter fields...")
    schema = get_filter_schema("ncl")
    if schema:
        print("✅ Available NCL filters:")
        print(f"   Filter fields (OR logic): {schema['filter_fields']}")
        print(f"   Must fields (AND logic): {schema['must_fields']}")
        print("=" * 60)
    else:
        print("❌ Could not retrieve schema")
    
    # Test search
    print("\n🔍 Testing search...")
    query = "日治時期，日本對台灣農產業帶來哪些影響"
    print(f"Query: {query}")
    print("=" * 60)
    
    results = search_documents(query, dataset="ncl", limit=3)
    print_results(results)

if __name__ == "__main__":
    main() 