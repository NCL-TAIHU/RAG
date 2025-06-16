from typing import Optional, List, Dict, ClassVar, Type, Self
from pydantic import BaseModel, create_model
from enum import Enum
from pymilvus import DataType
from dataclasses import dataclass
from src.core.document import Document, NCLDocument

class Filter(BaseModel):
    """
    Abstract base for statically declared filters.
    Subclasses should override `must_fields()` and `filter_fields()` manually.
    All fields should be `Optional[List[...]]` even if the underlying document field is singular.
    This unifies the semantics: all filters represent set-based inclusion or exclusion logic.
    """

    # The document class to validate against
    _doc_cls_: ClassVar[Optional[Type["Document"]]] = None

    @classmethod
    def must_fields(cls) -> List[str]:
        raise NotImplementedError("Subclasses must define their own must_fields.")

    @classmethod
    def filter_fields(cls) -> List[str]:
        raise NotImplementedError("Subclasses must define their own filter_fields.")
    
    def validate_fields(self, doc_cls: Type["Document"]) -> None:
        schema_fields = {f for f in doc_cls.metadata_schema()}
        declared_fields = set(self.filter_fields()) | set(self.must_fields())
        unknown = declared_fields - schema_fields
        if unknown:
            raise ValueError(f"{self.__name__} declares invalid fields: {unknown}")

    @classmethod
    def from_document_type(cls, doc_cls: Type["Document"]) -> Type["Filter"]:
        """
        Dynamically creates a subclass with fields from the doc's metadata schema.
        The subclass should still manually override `must_fields()` and `filter_fields()`.
        """
        fields = {
            f.name: (Optional[List[str]], None)
            for f in doc_cls.metadata_schema().values()
        }

        subclass_name = f"{doc_cls.__name__}Filter"

        # Dynamically create class, letting caller override must/filter fields
        return create_model(
            subclass_name,
            __base__=cls,
            **fields
        )

class NCLFilter(Filter.from_document_type(NCLDocument)):
    _doc_cls_ = NCLDocument  # triggers validate_fields during class creation

    def __init__(self, **data):
        super().__init__(**data)
        self.__class__.validate_fields(self._doc_cls_)

    @classmethod
    def filter_fields(cls) -> List[str]:
        return ["year", "category", "school_chinese", "dept_chinese"]

    @classmethod
    def must_fields(cls) -> List[str]:
        return ["keywords", "authors_chinese", "advisors_chinese"]