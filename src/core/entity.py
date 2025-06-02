from pydantic import BaseModel

class Document(BaseModel):
    """
    Represents a document with an ID and content.
    """
    id: str
    abstract: str
    content: str
    keywords: list[str] = []