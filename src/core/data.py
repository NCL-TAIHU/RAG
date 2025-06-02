from src.core.entity import Document
import os
from typing import Any, Iterator, List
import json
import yaml
config = yaml.safe_load(open("config/data.yml", "r"))

class DataLoader: 
    def load(self, *args, **kwargs) -> Iterator[List[Document]]:
        """
        Abstract method to load data.
        Should be implemented by subclasses.
        Iteratively returns batches of Documents.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    def from_default(cls, dataset: str, buffer_size: int = 64) -> "DataLoader":
        """
        Factory method to return a dataset-specific DataLoader.
        """
        if dataset == "history":
            base_path = config[dataset]['path']
            return PathsDataLoader(
                abstract_path=os.path.join(base_path, "abstracts"),
                content_path=os.path.join(base_path, "contents"),
                keywords_path=os.path.join(base_path, "keywords"),
                buffer_size=buffer_size
            )
        
        elif dataset == "arxiv":
            return JsonDataLoader(
                json_path=config[dataset]['path'],
                buffer_size=buffer_size
            )
        
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
class PathsDataLoader(DataLoader):
    def __init__(self, abstract_path: str, content_path: str, keywords_path: str,
                 MAX_FILES: int = None, buffer_size: int = 64):
        """
        Initializes the DataLoader with paths to the data files.
        """
        self.abstract_path = abstract_path
        self.content_path = content_path
        self.keywords_path = keywords_path
        self.MAX_FILES = MAX_FILES
        self.buffer_size = buffer_size
        self.documents: List[Document] = []

    def load(self):
        document_paths = os.listdir(self.abstract_path)
        content_paths = os.listdir(self.content_path)
        keywords_paths = os.listdir(self.keywords_path)

        # Use set intersection to find common files
        common_files = set(document_paths) & set(content_paths) & set(keywords_paths)
        print(f"Found {len(common_files)} common files in all directories.")

        for i, filename in enumerate(common_files):
            if self.MAX_FILES and i >= self.MAX_FILES:
                break
            with open(os.path.join(self.abstract_path, filename), 'r', encoding='utf-8') as f1, \
                 open(os.path.join(self.content_path, filename), 'r', encoding='utf-8') as f2, \
                 open(os.path.join(self.keywords_path, filename), 'r', encoding='utf-8') as f3:
                abstract = f1.read().strip()
                content = f2.read().strip()
                keywords = f3.read().strip().split(',')
                doc_id = os.path.splitext(filename)[0]
                yield from self.handle(Document(id=doc_id, abstract=abstract, content=content, keywords=keywords))
        yield from self.flush()

    def handle(self, document) -> Iterator[List[Document]]:
        #yields document batch if the buffer size is reached
        self.documents.append(document)
        if len(self.documents) >= self.buffer_size:
            yield from self.flush()

    def flush(self) -> Iterator[List[Document]]:
        # yields remaining documents if any
        if self.documents:
            yield self.documents
            self.documents = []

class JsonDataLoader(DataLoader):
    def __init__(self, json_path: str, buffer_size: int = 64):
        self.json_path = json_path
        self.buffer_size = buffer_size
        self.documents: List[Document] = []

    def load(self) -> Iterator[List[Document]]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                data = data.get("root", data)  # handle optional "root" wrapper
                doc = Document(
                    id=data["id"],
                    abstract=data.get("abstract", ""),
                    content=data.get("title", ""),
                    keywords=data.get("categories", "").split()
                )
                yield from self.handle(doc)
        yield from self.flush()


    def handle(self, document: Document) -> Iterator[List[Document]]:
        self.documents.append(document)
        if len(self.documents) >= self.buffer_size:
            yield from self.flush()

    def flush(self) -> Iterator[List[Document]]:
        if self.documents:
            yield self.documents
            self.documents = []
        
