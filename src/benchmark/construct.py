from pydantic import BaseModel
from typing import List
from src.core.llm import LLMBuilder, generate
from src.core.entity import Document
from src.core.data import Sampler, PrefixSampler, DataLoader
from src.benchmark.entity import Benchmark
from src.benchmark.io import save_benchmarks, load_benchmarks
from vllm import LLM
from tqdm import tqdm

CHATBOT = "meta-llama/Llama-3.1-8B-Instruct"
DATASET = "history"  # Default dataset to use``

def generate_benchmarks(documents: List[Document], prompt: str, llm: LLM) -> List[Benchmark]:
    '''
    Generates a sample question that each document can be an answer to.
    '''
    benchmarks = []
    for doc in tqdm(documents, desc="Generating benchmarks"):
        input_prompt = f"{prompt}\n\nDocument:\n{doc.abstract}"
        question = generate(llm, input_prompt)
        benchmarks.append(Benchmark(question=question, answer_id=doc.id))
    return benchmarks

if __name__ == "__main__":
    llm = LLMBuilder.from_default(CHATBOT).build()
    PROMPT = "請為以下文檔生成一個問題，該問題的答案應該是文檔的摘要。請確保問題是開放式的，並且能夠引導回答者提供詳細的答案。"
    dataloader = DataLoader.from_default(DATASET)
    sampler: Sampler = PrefixSampler(dataloader, max_samples = 100)
    documents = [doc for doc in sampler.sample()]
    benchmarks = generate_benchmarks(documents, PROMPT, llm)
    save_benchmarks(benchmarks, f"{DATASET}_benchmark.jsonl")