from typing import Optional, List, Dict, ClassVar, Type, Self, Any
from pydantic import BaseModel, create_model
from enum import Enum
from pymilvus import DataType
from dataclasses import dataclass
from datasets import load_dataset

# --- Enum for field types ---
class FieldType(str, Enum):
    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"

    def default_value(self):
        return {
            "str": "",
            "int": 0,
            "float": 0.0,
            "bool": False
        }[self.value]

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
    """
    A representation of a field in a document, used in both content and metadata.

    Attributes:
        name (str): The name of the field (e.g., "title", "abstract").
        contents (List[Any]): The values contained in the field (e.g., token list, raw strings).
        max_len (int): The maximum length constraint for this field (used for truncation or padding).
        type (FieldType): The type of the field, such as TEXT, KEYWORD, or TITLE.
    """
    name: str
    contents: List[Any] = []
    max_len: int
    type: FieldType
    def to_string(self) -> str:
        return f"{self.name}: {', '.join(map(str, self.contents))} ({self.type.value})"

# --- Abstract document interface ---
class Document:
    SCHEMA_INSTANCE: ClassVar[Self]

    def key(self) -> str:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Field]:
        raise NotImplementedError

    def channels(self) -> Dict[str, Field]:
        """
        A string that can be used to represent the content of the document.
        """
        raise NotImplementedError

    @classmethod
    def metadata_schema(cls) -> Dict[str, Field]:
        assert cls.SCHEMA_INSTANCE is not None, "Document schema instance is not initialized."
        return cls.SCHEMA_INSTANCE.metadata()

    @classmethod
    def channels_schema(cls) -> Dict[str, Field]:
        assert cls.SCHEMA_INSTANCE is not None, "Document schema instance is not initialized."
        return cls.SCHEMA_INSTANCE.channels()
    
    @classmethod
    def from_dataset(cls, dataset_name: str) -> Type["Document"]:
        """
        Factory method to gives the corresponding a Document class from a dataset.
        """
        if dataset_name == "ncl":
            return NCLDocument
        elif dataset_name == "litsearch":
            return LitSearchDocument
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are 'ncl' and 'litsearch'.")

# --------------NCL Document -----------------
# --- Info object for language-specific fields ---
class Info(BaseModel):
    """
    Language-specific information for a document.

    Attributes:
        title (Optional[str]): Title of the document in the specified language.
        school (Optional[str]): Name of the university or institution.
        dept (Optional[str]): Department or academic unit.
        abstract (Optional[str]): Abstract or summary text.
        authors (Optional[List[str]]): List of authors.
        advisors (Optional[List[str]]): List of advisors or supervisors.
    """
    title: Optional[str] = None
    school: Optional[str] = None
    dept: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    advisors: Optional[List[str]] = None

# --- NCLDocument concrete class ---
class NCLDocument(Document, BaseModel):
    """
    A document representation for the NCL (National Central Library) dataset.

    Attributes:
        SCHEMA_INSTANCE (ClassVar): Singleton instance used for schema binding or validation.
        id (str): Unique identifier for the document.
        year (Optional[int]): Graduation year of the thesis.
        category (Optional[str]): Degree category (e.g., Master's, PhD).
        chinese (Info): Language-specific metadata in Chinese.
        english (Info): Language-specific metadata in English.
        link (Optional[str]): URL link to the full thesis.
        keywords (List[str]): Keywords associated with the document.
    """

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

    def channels(self) -> Dict[str, Field]:
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

#------------ LitSearch Document --------------
class LitSearchDocument(Document, BaseModel):
    SCHEMA_INSTANCE: ClassVar["LitSearchDocument"]

    corpusid: int
    externalids: Dict[str, Optional[str]] = {}
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    pdfurl: Optional[str] = None

    def key(self) -> str:
        return str(self.corpusid)

    def metadata(self) -> Dict[str, Field]:
        data = [
            Field(name="corpusid", contents=[self.corpusid], type=FieldType.INTEGER, max_len=16),
            Field(name="year", contents=[self.year] if self.year else [], type=FieldType.INTEGER, max_len=4),
            Field(name="venue", contents=[self.venue] if self.venue else [], type=FieldType.STRING, max_len=128),
            Field(name="authors", contents=self.authors or [], type=FieldType.STRING, max_len=128),
            Field(name="doi", contents=[self.externalids.get("doi")] if self.externalids.get("doi") else [], type=FieldType.STRING, max_len=128),
            Field(name="arxiv", contents=[self.externalids.get("arxiv")] if self.externalids.get("arxiv") else [], type=FieldType.STRING, max_len=64),
            Field(name="dblp", contents=[self.externalids.get("dblp")] if self.externalids.get("dblp") else [], type=FieldType.STRING, max_len=128),
            Field(name="pdfurl", contents=[self.pdfurl] if self.pdfurl else [], type=FieldType.STRING, max_len=256)
        ]
        return {f.name: f for f in data}

    def channels(self) -> Dict[str, Field]:
        data = [
            Field(name="abstract", contents=[self.abstract] if self.abstract else [], type=FieldType.STRING, max_len=2048), 
            Field(name="title", contents=[self.title] if self.title else [], type=FieldType.STRING, max_len=256),
        ]
        return {f.name: f for f in data}

# --- Define the singleton schema instance ---
LitSearchDocument.SCHEMA_INSTANCE = LitSearchDocument(
    corpusid=0,
    title=None,
    abstract=None,
    authors=[],
    venue=None,
    year=None,
    pdfurl=None,
    externalids={}
)