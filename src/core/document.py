from typing import Optional, List, Dict, ClassVar, Type, Self, Any
from pydantic import BaseModel, create_model
from enum import Enum
from pymilvus import DataType
from dataclasses import dataclass

# --- Enum for field types ---
class FieldType(str, Enum):
    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"

    def to_python_type(self):
        return {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool
        }[self.value]

    def to_milvus_type(self):
        return {
            "str": DataType.VARCHAR,
            "int": DataType.INT64,
            "float": DataType.FLOAT,
            "bool": DataType.BOOL
        }[self.value]


# --- Field object used in both content and metadata ---
class Field(BaseModel):
    name: str
    contents: List[Any] = []
    max_len: int
    type: FieldType

# --- Abstract document interface ---
class Document:
    SCHEMA_INSTANCE: ClassVar[Self]

    def key(self) -> str:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Field]:
        raise NotImplementedError

    def content(self) -> Dict[str, Field]:
        raise NotImplementedError

    @classmethod
    def metadata_schema(cls) -> Dict[str, Field]:
        assert cls.SCHEMA_INSTANCE is not None, "Document schema instance is not initialized."
        return cls.SCHEMA_INSTANCE.metadata()

    @classmethod
    def content_schema(cls) -> Dict[str, Field]:
        assert cls.SCHEMA_INSTANCE is not None, "Document schema instance is not initialized."
        return [f.model_copy(update={"contents": []}) for f in cls.SCHEMA_INSTANCE.content()]

# --------------NCL Document -----------------
# --- Info object for language-specific fields ---
class Info(BaseModel):
    title: Optional[str] = None
    school: Optional[str] = None
    dept: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    advisors: Optional[List[str]] = None

# --- NCLDocument concrete class ---
class NCLDocument(Document, BaseModel):
    SCHEMA_INSTANCE: ClassVar["NCLDocument"]

    id: str
    year: Optional[int] = None
    category: Optional[str] = None
    chinese: Info
    english: Info
    link: Optional[str] = None
    keywords: List[str] = []

    def key(self) -> str:
        return self.id

    def metadata(self) -> Dict[str, Field]:
        data = [
            Field(name="year", contents=[self.year] if self.year else [], type=FieldType.INTEGER, max_len=4),
            Field(name="category", contents=[self.category] if self.category else [], type=FieldType.STRING, max_len=64),
            Field(name="link", contents=[self.link] if self.link else [], type=FieldType.STRING, max_len=256),
            Field(name="keywords", contents=self.keywords, type=FieldType.STRING, max_len=256),
            Field(name="school_chinese", contents=[self.chinese.school] if self.chinese.school else [], type=FieldType.STRING, max_len=128),
            Field(name="school_english", contents=[self.english.school] if self.english.school else [], type=FieldType.STRING, max_len=128),
            Field(name="dept_chinese", contents=[self.chinese.dept] if self.chinese.dept else [], type=FieldType.STRING, max_len=128),
            Field(name="dept_english", contents=[self.english.dept] if self.english.dept else [], type=FieldType.STRING, max_len=128),
            Field(name="authors_chinese", contents=self.chinese.authors or [], type=FieldType.STRING, max_len=256),
            Field(name="authors_english", contents=self.english.authors or [], type=FieldType.STRING, max_len=256),
            Field(name="advisors_chinese", contents=self.chinese.advisors or [], type=FieldType.STRING, max_len=256),
            Field(name="advisors_english", contents=self.english.advisors or [], type=FieldType.STRING, max_len=256),
        ]
        return {f.name: f for f in data}

    def content(self) -> Dict[str, Field]:
        data = [
            Field(name="abstract_chinese", contents=[self.chinese.abstract] if self.chinese.abstract else [], type=FieldType.STRING, max_len=1024),
            Field(name="abstract_english", contents=[self.english.abstract] if self.english.abstract else [], type=FieldType.STRING, max_len=1024),
            Field(name="title_chinese", contents=[self.chinese.title] if self.chinese.title else [], type=FieldType.STRING, max_len=256),
            Field(name="title_english", contents=[self.english.title] if self.english.title else [], type=FieldType.STRING, max_len=256),
        ]
        return {f.name: f for f in data}
    
# --- Define the singleton schema instance ---
NCLDocument.SCHEMA_INSTANCE = NCLDocument(
    id="",
    year=None,
    category=None,
    link=None,
    keywords=[],
    chinese=Info(),
    english=Info()
)