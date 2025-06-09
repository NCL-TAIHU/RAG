from src.benchmark.entity import Benchmark, Report
from typing import List
import json

def save_report(report: Report, file_path: str):
    '''
    Saves the evaluation report to a file in JSON format.
    '''
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report.model_dump(), f, indent=4, ensure_ascii=False)

def save_benchmarks(benchmarks: List[Benchmark], file_path: str):
    '''
    Saves the generated benchmarks to a file in JSONL format.
    Each line is a JSON representation of a Benchmark.
    '''
    with open(file_path, 'w', encoding='utf-8') as f:
        for b in benchmarks:
            json.dump(b.model_dump(), f, ensure_ascii=False)
            f.write('\n')


def load_benchmarks(file_path: str) -> List[Benchmark]:
    '''
    Loads benchmarks from a JSONL file.
    Each line is parsed into a Benchmark object.
    '''
    benchmarks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                benchmarks.append(Benchmark(**data))
    return benchmarks
  