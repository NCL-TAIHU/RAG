from pydantic import BaseModel
from typing import List, Optional

class Benchmark(BaseModel):
    question: str
    answer_ids: List[str]

class Report(BaseModel):
    """
    Represents the evaluation report for a search application.
    Contains metrics such as precision, recall, and F1 score.
    """
    top_k: int
    hits: int
    total: int
    description: Optional[str] = None