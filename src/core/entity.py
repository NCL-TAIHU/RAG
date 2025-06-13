from typing import Optional, List, Dict, ClassVar, Type, Self
from pydantic import BaseModel, create_model
from enum import Enum
from pymilvus import DataType


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
    contents: List[str] = []
    max_len: int
    type: FieldType


# --- Abstract document interface ---
class Document:
    SCHEMA_INSTANCE: ClassVar[Self]

    def __init_subclass__(cls):
        if not hasattr(cls, "SCHEMA_INSTANCE") or cls.SCHEMA_INSTANCE is None:
            raise TypeError(f"{cls.__name__} must define SCHEMA_INSTANCE")

    def key(self) -> str:
        raise NotImplementedError

    def metadata(self) -> List[Field]:
        raise NotImplementedError

    def content(self) -> List[Field]:
        raise NotImplementedError

    @classmethod
    def metadata_schema(cls) -> List[Field]:
        return [f.model_copy(update={"contents": []}) for f in cls.SCHEMA_INSTANCE.metadata()]

    @classmethod
    def content_schema(cls) -> List[Field]:
        return [f.model_copy(update={"contents": []}) for f in cls.SCHEMA_INSTANCE.content()]

class Filter(BaseModel):
    """
    Abstract base for statically declared filters.
    Subclasses should override `must_fields()` and `filter_fields()` manually.
    All fields should be `Optional[List[...]]` even if the underlying document field is singular.
    This unifies the semantics: all filters represent set-based inclusion or exclusion logic.
    """

    # The document class to validate against
    _doc_cls_: ClassVar[Optional[Type["Document"]]] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Run validation at subclass creation if a document class is bound
        if cls._doc_cls_ is not None:
            cls.validate_fields(cls._doc_cls_)

    @classmethod
    def must_fields(cls) -> List[str]:
        raise NotImplementedError("Subclasses must define their own must_fields.")

    @classmethod
    def filter_fields(cls) -> List[str]:
        raise NotImplementedError("Subclasses must define their own filter_fields.")
    
    @classmethod
    def validate_fields(cls, doc_cls: Type["Document"]) -> None:
        schema_fields = {f.name for f in doc_cls.metadata_schema()}
        declared_fields = set(cls.filter_fields()) | set(cls.must_fields())
        unknown = declared_fields - schema_fields
        if unknown:
            raise ValueError(f"{cls.__name__} declares invalid fields: {unknown}")

    @classmethod
    def from_document_type(cls, doc_cls: Type["Document"]) -> Type["Filter"]:
        """
        Dynamically creates a subclass with fields from the doc's metadata schema.
        The subclass should still manually override `must_fields()` and `filter_fields()`.
        """
        fields = {
            f.name: (Optional[List[str]], None)
            for f in doc_cls.metadata_schema()
        }

        subclass_name = f"{doc_cls.__name__}Filter"

        # Dynamically create class, letting caller override must/filter fields
        return create_model(
            subclass_name,
            __base__=cls,
            **fields
        )

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

    def metadata(self) -> List[Field]:
        return [
            Field(name="year", contents=[str(self.year)] if self.year else [], type=FieldType.INTEGER, max_len=4),
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

    def content(self) -> List[Field]:
        return [
            Field(name="abstract_chinese", contents=[self.chinese.abstract] if self.chinese.abstract else [], type=FieldType.STRING, max_len=1024),
            Field(name="abstract_english", contents=[self.english.abstract] if self.english.abstract else [], type=FieldType.STRING, max_len=1024),
            Field(name="title_chinese", contents=[self.chinese.title] if self.chinese.title else [], type=FieldType.STRING, max_len=256),
            Field(name="title_english", contents=[self.english.title] if self.english.title else [], type=FieldType.STRING, max_len=256),
        ]

class NCLFilter(Filter.from_document_type(NCLDocument)):
    _doc_cls_ = NCLDocument  # triggers validate_fields during class creation

    @classmethod
    def filter_fields(cls) -> List[str]:
        return ["year", "category", "school_chinese", "dept_chinese"]

    @classmethod
    def must_fields(cls) -> List[str]:
        return ["keywords", "authors_chinese", "advisors_chinese"]
    
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
