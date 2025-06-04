from pydantic import BaseModel

class Benchmark(BaseModel):
    question: str
    answer_id: str

class Report(BaseModel):
    """
    Represents the evaluation report for a search application.
    Contains metrics such as precision, recall, and F1 score.
    """
    top_k: int
    hits: int
    total: int