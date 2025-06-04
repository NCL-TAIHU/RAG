'''
evaluates a milvus collection with a list of queries and expected results.
The benchmark must be built on the same data as the collection, so that the correct answer exists in the collection.
'''
from app import SearchApp
from src.core.data import DataLoader, Sampler, DummySampler
from src.benchmark.io import load_benchmarks, save_report
from src.benchmark.entity import Benchmark, Report
import sys

#TODO: Implement Sampler
CHATBOT = "meta-llama/Llama-3.1-8B-Instruct"
DENSE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_EMBEDDER = "BAAI/bge-m3"
DATASET = "history"  # Default dataset to use
TOP_K = 10  # Default top_k value for search results
benchmarks_path = f"{DATASET}_benchmark.jsonl"  # Path to the benchmarks file

class SearchAppEvaluator: 
    '''
    Functions to evaluate the search application.
    Each function takes a SearchApp instance, a list of benchmarks, and a top_k value.
    '''
    @staticmethod
    def evaluate_dense(search_app: SearchApp, benchmarks: list[Benchmark], top_k) -> Report:
        """
        Evaluates the dense search functionality of the SearchApp.
        Returns a Report object with evaluation metrics.
        """
        hits = 0
        total = len(benchmarks)
        
        for benchmark in benchmarks:
            results = search_app.search(benchmark.question, method = "dense_search", limit=top_k)
            if any(result.id == benchmark.answer_id for result in results):
                hits += 1
        return Report(top_k=top_k, hits=hits, total=total)
    
    @staticmethod
    def evaluate_sparse(search_app: SearchApp, benchmarks: list[Benchmark], top_k) -> Report:
        """
        Evaluates the sparse search functionality of the SearchApp.
        Returns a Report object with evaluation metrics.
        """
        hits = 0
        total = len(benchmarks)
        
        for benchmark in benchmarks:
            results = search_app.search(benchmark.question, method = "sparse_search", limit=top_k)
            if any(result.id == benchmark.answer_id for result in results):
                hits += 1
        return Report(top_k=top_k, hits=hits, total=total)
    
    @staticmethod
    def evaluate_hybrid(search_app: SearchApp, benchmarks: list[Benchmark], top_k) -> Report:
        """
        Evaluates the hybrid search functionality of the SearchApp.
        Returns a Report object with evaluation metrics.
        """
        hits = 0
        total = len(benchmarks)
        
        for benchmark in benchmarks:
            results = search_app.search(benchmark.question, method = "hybrid_search", limit=top_k)  
            if any(result.id == benchmark.answer_id for result in results):
                hits += 1
        return Report(top_k=top_k, hits=hits, total=total)

if __name__ == "__main__":
    dataloader = DataLoader.from_default(DATASET)
    search_app = SearchApp(
        dataloader=dataloader,
        dense_embedder=DENSE_EMBEDDER, 
        sparse_embedder=SPARSE_EMBEDDER
    )
    search_app.setup()
    sampler = DummySampler(DATASET)
    documents = sampler.sample()
    try: 
        benchmarks = load_benchmarks(benchmarks_path)
    except FileNotFoundError:
        print(f"Benchmarks file {benchmarks_path} not found. Please run the benchmark construct.py for the target dataset.")
        sys.exit(1)  # Exit the program with non-zero exit code

    print(f"Loaded {len(benchmarks)} benchmarks from {benchmarks_path}")
    evaluator = SearchAppEvaluator()
    dense_report = evaluator.evaluate_dense(search_app, benchmarks, TOP_K)
    sparse_report = evaluator.evaluate_sparse(search_app, benchmarks, TOP_K)
    hybrid_report = evaluator.evaluate_hybrid(search_app, benchmarks, TOP_K)

    #save report
    save_report(dense_report, f"{DATASET}_dense_report.json")
    save_report(sparse_report, f"{DATASET}_sparse_report.json")
    save_report(hybrid_report, f"{DATASET}_hybrid_report.json")