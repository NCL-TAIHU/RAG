#!/usr/bin/env python3
"""
Comprehensive test runner for all preprocessors, document types, and data loaders
"""

import sys
import os
import json

# Add src to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.preprocess import NCLPreprocessor, PreprocessConfig, PreprocessorFactory, NCL_LLM_SummaryPreprocessor, LitSearchPreprocessor
from src.core.data import NCLDataLoader, NCL_LLM_SummaryDataLoader, LitSearchDataLoader
from src.core.document import NCLDocument, NCL_LLM_SummaryDocument, LitSearchDocument

def print_document_details(doc, doc_num: int = None):
    """Print detailed information about a document"""
    prefix = f"   Document {doc_num}: " if doc_num else "   "
    
    # Handle different document types with different ID fields
    if hasattr(doc, 'id'):
        doc_id = doc.id
    elif hasattr(doc, 'corpusid'):
        doc_id = doc.corpusid
    else:
        doc_id = "Unknown"
    
    print(f"{prefix}{doc_id}")
    print(f"     Document type: {type(doc).__name__}")
    
    # Print all document attributes
    for attr_name in dir(doc):
        if not attr_name.startswith('_') and not callable(getattr(doc, attr_name)):
            attr_value = getattr(doc, attr_name)
            if attr_value is not None:
                if isinstance(attr_value, str):
                    # Truncate long strings
                    display_value = attr_value[:100] if len(attr_value) > 100 else attr_value
                    print(f"     {attr_name}: {display_value}")
                elif isinstance(attr_value, list):
                    # Show list items
                    if attr_value:
                        display_value = str(attr_value[:3]) if len(attr_value) > 3 else str(attr_value)
                        print(f"     {attr_name}: {display_value}")
                elif isinstance(attr_value, dict):
                    # Show dictionary keys
                    if attr_value:
                        print(f"     {attr_name}: {list(attr_value.keys())}")
                else:
                    print(f"     {attr_name}: {attr_value}")
    
    # Special handling for nested objects
    if hasattr(doc, 'chinese') and doc.chinese:
        print(f"     Chinese Info:")
        chinese = doc.chinese
        for attr_name in ['title', 'school', 'dept', 'abstract', 'authors', 'advisors']:
            if hasattr(chinese, attr_name):
                attr_value = getattr(chinese, attr_name)
                if attr_value:
                    if isinstance(attr_value, str):
                        display_value = attr_value[:200] if len(attr_value) > 200 else attr_value
                        print(f"       {attr_name}: {display_value}")
                    elif isinstance(attr_value, list):
                        if attr_value:
                            display_value = str(attr_value[:3]) if len(attr_value) > 3 else str(attr_value)
                            print(f"       {attr_name}: {display_value}")
    
    if hasattr(doc, 'english') and doc.english:
        print(f"     English Info:")
        english = doc.english
        for attr_name in ['title', 'school', 'dept', 'abstract', 'authors', 'advisors']:
            if hasattr(english, attr_name):
                attr_value = getattr(english, attr_name)
                if attr_value:
                    if isinstance(attr_value, str):
                        display_value = attr_value[:80] if len(attr_value) > 80 else attr_value
                        print(f"       {attr_name}: {display_value}")
                    elif isinstance(attr_value, list):
                        if attr_value:
                            display_value = str(attr_value[:3]) if len(attr_value) > 3 else str(attr_value)
                            print(f"       {attr_name}: {display_value}")
    
    print()  # Add spacing between documents

def test_preprocessor(preprocessor_type: str, file_path: str, max_records: int = 3):
    """Test a specific preprocessor type"""
    print(f"\n{'='*60}")
    print(f"Testing {preprocessor_type.upper()} Preprocessor")
    print(f"{'='*60}")
    
    try:
        # Create preprocessor
        print(f"\n1. Creating {preprocessor_type} preprocessor...")
        config = PreprocessConfig(input_path=os.path.dirname(file_path))
        preprocessor = PreprocessorFactory.create(preprocessor_type, config)
        print(f"✓ {preprocessor_type} preprocessor created successfully")
        
        # Test field mappings
        print(f"\n2. Field mappings for {preprocessor_type}:")
        mappings = preprocessor.field_mappings
        print(f"✓ Generated {len(mappings)} field mappings:")
        for source, target in mappings.items():
            print(f"   {source} -> {target}")
        
        # Test processing records
        print(f"\n3. Processing {max_records} sample records...")
        processed_count = 0
        error_count = 0
        documents = []  # Store documents for detailed printing
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_records:
                    break
                
                if line.strip():
                    try:
                        raw_data = json.loads(line)
                        print(f"\n   Processing record {i+1}:")
                        print(f"     Raw data keys: {list(raw_data.keys())}")
                        
                        # Test each step
                        extracted = preprocessor.extract_fields(raw_data)
                        print(f"     Extracted fields: {list(extracted.keys())}")
                        content = preprocessor.fill_content(extracted)
                        print(f"     Content structure: {list(content.keys())}")
                        document = preprocessor.preprocess_data(content)
                        
                        if document:
                            processed_count += 1
                            documents.append(document)
                            print(f"     ✓ Success: {document.id}")
                            print(f"       Document type: {type(document).__name__}")
                        else:
                            error_count += 1
                            print(f"     ❌ Failed: Invalid data (filtered out)")
                            
                    except Exception as e:
                        error_count += 1
                        print(f"     ❌ Error: {e}")
        
        print(f"\n✓ Processed {processed_count} valid records, {error_count} errors")
        
        # Print detailed document information
        if documents:
            print(f"\n4. Detailed Document Information:")
            print(f"{'='*60}")
            for i, doc in enumerate(documents, 1):
                print_document_details(doc, i)
        
        return processed_count > 0
        
    except Exception as e:
        print(f"❌ {preprocessor_type} test failed: {e}")
        return False

def test_data_loader(loader_type: str, file_path: str, max_docs: int = 3):
    """Test a specific data loader type"""
    print(f"\n{'='*60}")
    print(f"Testing {loader_type.upper()} Data Loader")
    print(f"{'='*60}")
    
    try:
        # Create appropriate loader
        if loader_type == "ncl":
            loader = NCLDataLoader(file_path, buffer_size=2)
        elif loader_type == "ncl_llm_summary":
            loader = NCL_LLM_SummaryDataLoader(file_path, buffer_size=2)
        elif loader_type == "litsearch":
            loader = LitSearchDataLoader(buffer_size=2)
        else:
            print(f"❌ Unknown loader type: {loader_type}")
            return False
        
        print(f"✓ {loader_type} data loader created successfully")
        
        # Test streaming
        print(f"\n1. Testing stream() method...")
        total_docs = 0
        documents = []  # Store documents for detailed printing
        
        for doc in loader.stream():
            total_docs += 1
            if total_docs <= max_docs:
                documents.append(doc)
                print(f"   ✓ Document {total_docs}: {doc.id}")
                print(f"     Document type: {type(doc).__name__}")
            if total_docs >= max_docs:
                break
        
        print(f"✓ Stream found {total_docs} documents")
        
        # Print detailed document information
        if documents:
            print(f"\n2. Detailed Document Information:")
            print(f"{'='*60}")
            for i, doc in enumerate(documents, 1):
                print_document_details(doc, i)
        
        # Test batch loading
        print(f"\n3. Testing load() method...")
        batch_count = 0
        total_in_batches = 0
        for batch in loader.load():
            batch_count += 1
            total_in_batches += len(batch)
            if total_in_batches >= max_docs:
                break
        print(f"✓ Created {batch_count} batches with {total_in_batches} total documents")
        
        return True
        
    except Exception as e:
        print(f"❌ {loader_type} loader test failed: {e}")
        return False

def test_with_real_data():
    """Test all preprocessors and data loaders with real data"""
    print("=== Comprehensive Testing of All Preprocessors and Data Loaders ===\n")
    
    # Your data file path
    file_path = "/home/wenjieluu1130/RAG/data/test/108.jsonl"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📁 Testing with file: {file_path}")
    
    # Test all preprocessors
    preprocessor_types = ["ncl", "ncl_llm_summary"]
    preprocessor_results = {}
    
    for preprocessor_type in preprocessor_types:
        success = test_preprocessor(preprocessor_type, file_path, max_records=3)
        preprocessor_results[preprocessor_type] = success
    
    # Test all data loaders
    loader_types = ["ncl", "ncl_llm_summary"]
    loader_results = {}
    
    for loader_type in loader_types:
        success = test_data_loader(loader_type, file_path, max_docs=3)
        loader_results[loader_type] = success
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    print("\nPreprocessor Results:")
    for preprocessor_type, success in preprocessor_results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"   {preprocessor_type}: {status}")
    
    print("\nData Loader Results:")
    for loader_type, success in loader_results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"   {loader_type}: {status}")
    
    # Overall success
    all_preprocessors_passed = all(preprocessor_results.values())
    all_loaders_passed = all(loader_results.values())
    
    if all_preprocessors_passed and all_loaders_passed:
        print(f"\n🎉 All tests completed successfully!")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")

def test_litsearch_preprocessor():
    """Test LitSearch preprocessor (requires different data source)"""
    print(f"\n{'='*60}")
    print("Testing LITSEARCH Preprocessor")
    print(f"{'='*60}")
    
    try:
        print("\n1. Creating LitSearch preprocessor...")
        config = PreprocessConfig(input_path="dummy_path")  # LitSearch doesn't use file path
        preprocessor = PreprocessorFactory.create("litsearch", config)
        print("✓ LitSearch preprocessor created successfully")
        
        # Test with sample data
        print("\n2. Testing with sample LitSearch data...")
        sample_data = {
            "corpusid": 12345,
            "externalids": {"doi": "10.1234/test"},
            "title": "Sample Research Paper",
            "abstract": "This is a sample abstract for testing.",
            "authors": ["Author 1", "Author 2"],
            "venue": "Sample Conference",
            "year": 2023,
            "pdfurl": "https://example.com/paper.pdf"
        }
        
        extracted = preprocessor.extract_fields(sample_data)
        print(f"     Extracted fields: {extracted}")
        
        content = preprocessor.fill_content(extracted)
        print(f"     Content: {content}")
        
        document = preprocessor.preprocess_data(content)
        if document:
            print(f"     ✓ Success: {document.corpusid}")
            print(f"       Title: {document.title}")
            print(f"       Abstract: {document.abstract[:50]}")
            print(f"       Authors: {document.authors}")
            
            # Print detailed document information
            print(f"\n3. Detailed Document Information:")
            print(f"{'='*60}")
            print_document_details(document)
        else:
            print(f"     ❌ Failed: Invalid data")
        
        return True
        
    except Exception as e:
        print(f"❌ LitSearch test failed: {e}")
        return False

def test_litsearch_loader():
    """Test LitSearch data loader"""
    print(f"\n{'='*60}")
    print("Testing LITSEARCH Data Loader")
    print(f"{'='*60}")
    
    try:
        print("\n1. Creating LitSearch data loader...")
        loader = LitSearchDataLoader(buffer_size=2)
        print("✓ LitSearch data loader created successfully")
        
        print("\n2. Testing stream() method (first 3 documents)...")
        total_docs = 0
        documents = []  # Store documents for detailed printing
        
        for doc in loader.stream():
            total_docs += 1
            if total_docs <= 3:
                documents.append(doc)
                print(f"   ✓ Document {total_docs}: {doc.corpusid}")
                print(f"     Title: {doc.title[:50] if doc.title else 'None'}")
                print(f"     Abstract: {doc.abstract[:50] if doc.abstract else 'None'}")
            if total_docs >= 3:
                break
        
        print(f"✓ Stream found {total_docs} documents")
        
        # Print detailed document information
        if documents:
            print(f"\n3. Detailed Document Information:")
            print(f"{'='*60}")
            for i, doc in enumerate(documents, 1):
                print_document_details(doc, i)
        
        return True
        
    except Exception as e:
        print(f"❌ LitSearch loader test failed: {e}")
        return False

if __name__ == '__main__':
    # Test NCL-based preprocessors and loaders
    test_with_real_data()
    
    # Test LitSearch preprocessor and loader
    test_litsearch_preprocessor()
    test_litsearch_loader()
    
    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETED")
    print(f"{'='*80}")