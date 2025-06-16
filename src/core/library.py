from src.core.document import Document
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)
class Library: 
    '''
    abstract class. 
    stores and retrieves documents based on their IDs.
    '''
    def clear(self) -> None: 
        """
        Clears the library by removing all documents.
        This method should be overridden by subclasses if needed.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def insert(self, docs: List[Document]) -> None:
        """
        Inserts a list of documents into the library.
        :param docs: A list of Document objects to be inserted.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def retrieve(self, ids: List[str]) -> List[Document]:
        """
        Retrieves a list of documents from the library based on their IDs.
        :param ids: A list of document IDs to retrieve.
        :return: A list of Document objects corresponding to the provided IDs.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    

class InMemoryLibrary(Library):
    """
    In-memory library for storing and retrieving documents.
    """
    def __init__(self, file_path: str = "in_memory_library.json"):
        self.documents: Dict[str, Document] = {}
        self.file_path = file_path

    def insert(self, docs: List[Document]) -> None:
        for doc in docs:
            self.documents[doc.key()] = doc

    def retrieve(self, ids: List[str]) -> List[Document]:
        return [self.documents[id_] for id_ in ids if id_ in self.documents]
    
    def clear(self) -> None:
        """
        Clears the in-memory library by removing all documents.
        """
        self.documents.clear()
        logger.info("In-memory library cleared.")
    
class FilesLibrary(Library):
    """
    File-based library for storing and retrieving documents. Retrieves at query time from disk. 
    """
    def __init__(self, base_path: str):
        self.base_path = base_path


    def insert(self, docs: List[Document]) -> None:
        for doc in docs: 
            file_path = f"{self.base_path}/{doc.id}.json"
            with open(file_path, 'w') as f:
                json.dump(doc.model_dump(), f)
    
    def retrieve(self, ids: List[str]) -> List[Document]:
        documents = []
        for id_ in ids:
            file_path = f"{self.base_path}/{id_}.json"
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    documents.append(Document(**data))
            except FileNotFoundError:
                logger.warning(f"Document with ID {id_} not found in {self.base_path}.")
        return documents
    
    def clear(self) -> None:
        """
        Clears the file-based library by removing all documents.
        This method deletes all files in the base path.
        """
        import os
        for filename in os.listdir(self.base_path):
            file_path = os.path.join(self.base_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")
        logger.info("File-based library cleared.")