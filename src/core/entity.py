from pydantic import BaseModel

class MetaData(BaseModel):
    """
    Represents metadata for a document, including an ID and optional tags.
    """
    id: str
    content: str
    keywords: list[str] = []

class Document(BaseModel):
    """
    Represents a document with an ID and content.
    """
    id: str
    abstract: str