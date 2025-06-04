from pydantic import BaseModel
from typing import List
from src.core.llm import LLMBuilder, generate
from src.core.entity import Document
from src.core.data import Sampler, DummySampler
from src.benchmark.entity import Benchmark
from src.benchmark.io import save_benchmarks, load_benchmarks
from vllm import LLM

CHATBOT = "meta-llama/Llama-3.1-8B-Instruct"
DATASET = "history"  # Default dataset to use``

def generate_benchmarks(documents: List[Document], prompt: str, llm: LLM) -> List[Benchmark]:
    '''
    Generates a sample question that each document can be an answer to.
    '''
    benchmarks = []
    for doc in documents:
        input_prompt = f"{prompt}\n\nDocument:\n{doc.abstract}"
        question = generate(llm, input_prompt)
        benchmarks.append(Benchmark(question=question, answer_id=doc.id))
    return benchmarks

if __name__ == "__main__":
    llm = LLMBuilder.from_default(CHATBOT).build()
    PROMPT = "Generate a question that can be answered by the following document."
    sampler: Sampler = DummySampler(DATASET)
    documents = sampler.sample()
    benchmarks = generate_benchmarks(documents, PROMPT, llm)
    save_benchmarks(benchmarks, f"{DATASET}_benchmark.jsonl")

