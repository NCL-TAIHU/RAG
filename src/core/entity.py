from pydantic import BaseModel
from typing import Optional, List

class Info(BaseModel): 
    title: Optional[str] = None
    school: Optional[str] = None
    dept: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    advisors: Optional[List[str]] = None
    
class Document(BaseModel):
    """
    Represents a document with an ID and content.
    """
    id: str
    year: Optional[int] = None
    category: Optional[str] = None 
    chinese: Info
    english: Info
    link: Optional[str]
    keywords: list[str] = []