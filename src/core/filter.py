from typing import Optional, List, Dict, ClassVar, Type, Self
from pydantic import BaseModel, create_model
from enum import Enum
from pymilvus import DataType
from dataclasses import dataclass
from src.core.document import Document, NCLDocument, LitSearchDocument, NCL_LLM_SummaryDocument

class Filter(BaseModel):
    """
    Base class for statically declared filters.
    Filters are dependent on specific document types and are used to filter search results.
    Subclasses should override `must_fields()` and `filter_fields()` manually.
    All fields should be `Optional[List[...]]` even if the underlying document field is singular.
    This unifies the semantics: all filters represent set-based inclusion or exclusion logic.
    """
    # The document class to validate against
    _doc_cls_: ClassVar[Optional[Type["Document"]]] = None
    EMPTY: ClassVar["Filter"]

    def set_fields(self, **kwargs):
        """
        Set fields dynamically based on the provided keyword arguments.
        This allows for flexible filter creation.
        """
        for field in kwargs:
            if field not in self.filter_fields() and field not in self.must_fields():
                raise ValueError(f"Invalid field: {field}. Must be one of {self.filter_fields() + self.must_fields()}")
            setattr(self, field, kwargs[field])

        return self

    @classmethod
    def must_fields(cls) -> List[str]:
        '''
        Must fields must all be present in the document.
        '''
        raise NotImplementedError("Subclasses must define their own must_fields.")

    @classmethod
    def filter_fields(cls) -> List[str]:
        '''
        Filter fields have to be at least one present in the document.
        '''
        raise NotImplementedError("Subclasses must define their own filter_fields.")
    
    @classmethod
    def validate_fields(cls, doc_cls: Type["Document"]) -> None:
        schema_fields = {f.name for f in doc_cls.metadata_schema().values()}
        declared_fields = set(cls.filter_fields()) | set(cls.must_fields())
        unknown = declared_fields - schema_fields
        if unknown:
            raise ValueError(f"{cls.__name__} declares invalid fields: {unknown}")


    @classmethod
    def from_document_type(cls, doc_cls: Type["Document"]) -> Type["Filter"]:
        """
        Used Internally to create a filter class based on a Document type.
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
    
    @classmethod
    def from_dataset(cls, dataset_name: str) -> Type["Filter"]:
        """
        Creates a filter class based on a HuggingFace dataset.
        The dataset must have a metadata schema compatible with Document.
        """
        if dataset_name == "ncl":
            return NCLFilter
        elif dataset_name == "litsearch":
            return LitSearchFilter
        elif dataset_name == "ncl_llm_summary":
            return NCL_LLM_SummaryFilter
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets are 'ncl', 'litsearch', and 'ncl_llm_summary'.")
        
class NCLFilter(Filter.from_document_type(NCLDocument)):
    _doc_cls_ = NCLDocument  # triggers validate_fields during class creation
    EMPTY: ClassVar["NCLFilter"]

    
    def __init__(self, **data):
        super().__init__(**data)
        self.__class__.validate_fields(self._doc_cls_)

    @classmethod
    def filter_fields(cls) -> List[str]:
        return ["year", "category", "school_chinese", "dept_chinese"]

    @classmethod
    def must_fields(cls) -> List[str]:
        return ["keywords", "authors_chinese", "advisors_chinese"]
    
NCLFilter.EMPTY = NCLFilter()
    
class LitSearchFilter(Filter.from_document_type(LitSearchDocument)):
    _doc_cls_ = LitSearchDocument
    EMPTY: ClassVar["LitSearchFilter"]

    def __init__(self, **data):
        super().__init__(**data)
        self.__class__.validate_fields(self._doc_cls_)

    @classmethod
    def filter_fields(cls) -> List[str]:
        return ["year", "venue"]

    @classmethod
    def must_fields(cls) -> List[str]:
        return ["authors"]
    
LitSearchFilter.EMPTY = LitSearchFilter()

class NCL_LLM_SummaryFilter(Filter.from_document_type(NCL_LLM_SummaryDocument)):
    _doc_cls_ = NCL_LLM_SummaryDocument
    EMPTY: ClassVar["NCL_LLM_SummaryFilter"]

    def __init__(self, **data):
        super().__init__(**data)
        self.__class__.validate_fields(self._doc_cls_)

    @classmethod
    def filter_fields(cls) -> List[str]:
        return ["year", "category", "school_chinese", "dept_chinese"]

    @classmethod
    def must_fields(cls) -> List[str]:
        return ["keywords", "authors_chinese", "advisors_chinese"]
    
NCL_LLM_SummaryFilter.EMPTY = NCL_LLM_SummaryFilter()