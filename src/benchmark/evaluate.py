'''
evaluates a milvus collection with a list of queries and expected results.
The benchmark must be built on the same data as the collection, so that the correct answer exists in the collection.
'''
from src.main.app import SearchApp
from src.core.data import DataLoader, Sampler, PrefixSampler
from src.benchmark.io import BenchmarkFactory, save_report
from src.benchmark.entity import Benchmark, Report
from src.core.library import Library, InMemoryLibrary
from src.core.search_engine import HybridSearchEngine, MilvusSearchEngine, ElasticSearchEngine
from src.core.embedder import SparseEmbedder, DenseEmbedder, BGEM3Embedder, AutoModelEmbedder
from src.core.manager import Manager
from src.core.document import Document
from src.core.filter import Filter
from typing import Iterator
import sys

#TODO: Implement Sampler
CHATBOT = "meta-llama/Llama-3.1-8B-Instruct"
DENSE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_EMBEDDER = "BAAI/bge-m3"
DATASET = "litsearch"  # Default dataset to use
BENCHMARK: str = "litsearch"  # Default benchmark source
TOP_K = 10  # Default top_k value for search results
benchmarks_path = f"{DATASET}_benchmark.jsonl"  # Path to the benchmarks file

class SearchAppEvaluator: 
    '''
    Functions to evaluate the search application.
    Each function takes a SearchApp instance, a list of benchmarks, and a top_k value.
    '''
    
    @staticmethod
    def evaluate_hybrid(search_app: SearchApp, benchmarks: Iterator[Benchmark], top_k) -> Report:
        """
        Evaluates the hybrid search functionality of the SearchApp.
        Returns a Report object with evaluation metrics.
        """
        hits = 0
        total = 0
        filt = Filter.from_dataset(DATASET).EMPTY
        for benchmark in benchmarks:
            results = search_app.search(benchmark.question, filt, limit=top_k)  
            if any((result.key() in benchmark.answer_ids) for result in results):
                hits += 1
            total += 1
        return Report(top_k=top_k, hits=hits, total=total)

if __name__ == "__main__":
    DOC_CLS = Document.from_dataset(DATASET)  # Default document class based on dataset
    FILT_CLS = Filter.from_dataset(DATASET)  # Default filter class based on dataset
    dataloader = DataLoader.from_default(DATASET)
    library: Library = InMemoryLibrary()
    sparse_embedder: SparseEmbedder = BGEM3Embedder(model_name=SPARSE_EMBEDDER)
    dense_embedder: DenseEmbedder = AutoModelEmbedder(model_name=DENSE_EMBEDDER)
    engine1 = HybridSearchEngine(
        relational_search_engine=ElasticSearchEngine("https://localhost:9200", document_cls=DOC_CLS, filter_cls=FILT_CLS, es_index= "documents"),
        vector_search_engine=MilvusSearchEngine(sparse_embedder, dense_embedder, document_cls=DOC_CLS, filter_cls=FILT_CLS)
    )
    engine2 = MilvusSearchEngine(sparse_embedder, dense_embedder, document_cls=DOC_CLS, filter_cls=FILT_CLS)
    manager = Manager(library, [engine1, engine2], router_name="sparsity")
    app = SearchApp(dataloader, manager, max_files=10000000)
    app.setup()
    factory = BenchmarkFactory.from_default(BENCHMARK)
    evaluator = SearchAppEvaluator()
    for k in list(range(1, 11)) + list(range(20, 101, 10)):
        hybrid_report = evaluator.evaluate_hybrid(app, factory.stream(), k)
        hybrid_report.description = ("Content schema: \n" + '\n'.join([f.to_string() for f in DOC_CLS.content_schema().values()]) +
                                    "\n Metadata schema: \n" + "\n".join([f.to_string() for f in DOC_CLS.metadata_schema().values()])  +
                                    "\n Filter must fields: \n" + '\n'.join(FILT_CLS.must_fields()) +
                                    "\n Filter filter fields: \n" + '\n'.join(FILT_CLS.filter_fields()))  
        
        save_report(hybrid_report, f"reports/{DATASET}_top{k}_hybrid_report.json")