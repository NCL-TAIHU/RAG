from pydantic import BaseModel
from typing import List
from src.core.llm import Agent
from src.core.document import Document
from src.core.data import Sampler, PrefixSampler, DataLoader
from src.benchmark.entity import Benchmark
from src.benchmark.io import save_benchmarks, load_benchmarks
from tqdm import tqdm

CHATBOT = "taide/Llama-3.1-TAIDE-LX-8B-Chat" #"taide/TAIDE-LX-7B" #"meta-llama/Llama-3.1-8B-Instruct" #
DATASET = "ncl"  # Default dataset to use``

def generate_benchmarks(documents: List[Document], prompt: str, llm: Agent) -> List[Benchmark]:
    '''
    Generates a sample question that each document can be an answer to.
    '''
    benchmarks = []
    for doc in tqdm(documents, desc="Generating benchmarks"):
        input_prompt = f"{prompt}\n\nDocument:\n{doc.channels().values()[0]}"
        question = llm.generate(input_prompt).strip()
        benchmarks.append(Benchmark(question=question, answer_ids=[doc.id]))
    return benchmarks

if __name__ == "__main__":
    llm = Agent.from_vllm(CHATBOT)
    PROMPT = "請為以下文檔生成一個問題，該問題的答案應該是文檔的摘要。請給問體本身就好。"
    dataloader = DataLoader.from_default(DATASET)
    sampler: Sampler = PrefixSampler(dataloader, max_samples = 100)
    documents = [doc for doc in sampler.sample()]
    benchmarks = generate_benchmarks(documents, PROMPT, llm)
    save_benchmarks(benchmarks, f"{DATASET}_benchmark.jsonl")