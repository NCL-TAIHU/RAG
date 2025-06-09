from pydantic import BaseModel
from typing import List
from src.core.llm import LLMBuilder, generate
from src.core.entity import Document
from src.core.data import DataLoader, Sampler, DummySampler
from src.benchmark.entity import Benchmark
from src.benchmark.io import save_benchmarks, load_benchmarks
from vllm import LLM
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

CHATBOT = "o3-mini"
USE_OPENAI = True  
DATASET = "history"  # Default dataset to use``

def generate_benchmarks(documents: List[Document], prompt: str, llm: LLM) -> List[Benchmark]:
    '''
    Generates a sample question that each document can be an answer to.
    '''
    benchmarks = []
    for doc in tqdm(documents):
        input_prompt = f"{doc.keywords}\n\n{doc.abstract}\n\n{prompt}"
        response = generate(llm, input_prompt)
        question = response[0]["text"] if response else "No response generated."
        benchmarks.append(Benchmark(question=question, answer_id=doc.id))
    return benchmarks

if __name__ == "__main__":
    llm = LLMBuilder.from_default(CHATBOT, use_openai=USE_OPENAI).build()
        
    PROMPT ="""
        當你提出什麼樣的問題時，會期待這個基於RAG技術的論文檢索系統幫你檢索出以上論文？
        問題的敘述方式是在還沒看過這篇論文的情況下會給出的提問，所以不會出現以下有“論文中...”或“該論文...”等字眼的問題。
        例如：
        這篇論文主要探討楚地哪些方面的生活和風俗現象？ 
        該論文涵蓋了哪個歷史時期的楚地風俗？ 
        作者如何詮釋日書在古代楚地民間風俗中的文化意涵？   

        只需給我一個問題，且給我問題本身就好，不要多加解釋。"""
    #PROMPT = "Generate a question that can be answered by the following document."
    dataloader = DataLoader.from_default(DATASET)
    #sampler: Sampler = DummySampler(DATASET)
    sampler = Sampler(dataloader, n=20)
    documents = sampler.sample()
    # print(len(documents))
    # print(type(documents[0]))
    # input()
    benchmarks = generate_benchmarks(documents, PROMPT, llm)
    save_benchmarks(benchmarks, f"{DATASET}_benchmark.jsonl")