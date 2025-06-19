from src.benchmark.entity import Benchmark, Report
from typing import List, Iterator
import json
from datasets import load_dataset

def save_report(report: Report, file_path: str):
    '''
    Saves the evaluation report to a file in JSON format.
    '''
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

def save_benchmarks(benchmarks: List[Benchmark], file_path: str):
    '''
    Saves the generated benchmarks to a file in JSONL format.
    Each line is a JSON representation of a Benchmark.
    '''
    with open(file_path, 'w', encoding='utf-8') as f:
        for b in benchmarks:
            json.dump(b.model_dump(), f, ensure_ascii=False)
            f.write('\n')


class BenchmarkFactory:
    def __init__(self, source: str, path: str = None):
        self.source = source
        self.path = path

    @classmethod
    def from_default(cls, name: str, path: str = None) -> "BenchmarkFactory":
        """
        Factory method to construct a BenchmarkFactory.
        - name="auto" expects a local JSONL file path.
        - name="litsearch" loads from HuggingFace datasets.
        """
        if name == "auto" and not path:
            raise ValueError("Path must be provided for 'auto' benchmark source.")
        if name not in {"auto", "litsearch"}:
            raise ValueError(f"Unsupported benchmark source: {name}")
        return cls(name, path)

    def stream(self) -> Iterator[Benchmark]:
        """
        Yields Benchmark objects from the configured source.
        """
        if self.source == "auto":
            yield from self._stream_local()
        elif self.source == "litsearch":
            yield from self._stream_litsearch()
        else:
            raise NotImplementedError(f"Unsupported source: {self.source}")

    def _stream_local(self) -> Iterator[Benchmark]:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    yield Benchmark(**data)

    def _stream_litsearch(self) -> Iterator[Benchmark]:
        query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
        for entry in query_data:
            yield Benchmark(
                question=entry["query"],
                answer_ids=[str(cid) for cid in entry.get("corpusids", [])]
            )