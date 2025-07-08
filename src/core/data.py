from src.core.document import Document, NCLDocument, Info, LitSearchDocument
import os
from typing import Any, Iterator, List, Dict
import json
import yaml
from datasets import load_dataset
from tqdm import tqdm
from src.core.util import coalesce
import uuid

class DataLoader: 
    def load(self, *args, **kwargs) -> Iterator[List[Document]]:
        """
        Abstract method to load data.
        Should be implemented by subclasses.
        Iteratively returns batches of Documents.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def stream(self) -> Iterator[Document]:
        """
        Abstract method to stream data.
        Should be implemented by subclasses.
        Iteratively returns Documents one by one.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @classmethod
    def from_default(cls, dataset: str, buffer_size: int = 64) -> "DataLoader":
        """
        Factory method to return a dataset-specific DataLoader.
        """
        config = yaml.safe_load(open("config/data.yml", "r"))
        if dataset == "history":
            raise NotImplementedError("Arxiv dataset loader is not implemented yet.")
            base_path = config[dataset]['path']
            return PathsDataLoader(
                abstract_path=os.path.join(base_path, "abstracts"),
                content_path=os.path.join(base_path, "contents"),
                keywords_path=os.path.join(base_path, "keywords"),
                buffer_size=buffer_size
            )
        
        elif dataset == "ncl": 
            return NCLDataLoader(
                base_path=config[dataset]['path'],
                buffer_size=buffer_size
            )
        
        elif dataset == 'litsearch':
            return LitSearchDataLoader(
                buffer_size=buffer_size
            )
        
        elif dataset == "arxiv":
            raise NotImplementedError("Arxiv dataset loader is not implemented yet.")
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

    def stream(self) -> Iterator[Document]: 
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
                yield Document(id=doc_id, abstract=abstract, content=content, keywords=keywords)

    def load(self):
        for doc in self.stream(): 
            yield from self._handle(doc)
        yield from self._flush()

    def _handle(self, document) -> Iterator[List[Document]]:
        #yields document batch if the buffer size is reached
        self.documents.append(document)
        if len(self.documents) >= self.buffer_size:
            yield from self._flush()

    def _flush(self) -> Iterator[List[Document]]:
        # yields remaining documents if any
        if self.documents:
            yield self.documents
            self.documents = []

class NCLDataLoader(DataLoader):
    def __init__(self, base_path: str, buffer_size: int = 64):
        """
        Initializes the NCLDataLoader.

        Args:
            base_path (str): Path to the directory containing .jsonl files.
            buffer_size (int, optional): Number of documents per batch. Defaults to 64.
        """
        self.base_path = base_path
        self.json_paths = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.jsonl')]
        self.buffer_size = buffer_size
        self.documents: List[Document] = []

    def stream(self) -> Iterator[Document]:
        counter = 0
        for path in self.json_paths: 
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data: Dict = json.loads(line)
                    yield NCLDocument(
                        id=data['uid'],
                        year=data.get("畢業學年度", None), 
                        category=data.get("學位類別", None),
                        chinese=Info(
                            title=data.get("論文名稱", None),
                            school=data.get("學校名稱", None),
                            dept=data.get("系所名稱", None),
                            abstract=data.get("摘要", None),
                            authors=data.get("作者", "").splitlines(),
                            advisors=data.get("指導教授", "").splitlines()
                        ),
                        english=Info(
                            title=data.get("論文名稱(外文)", None),
                            school=data.get("學校名稱(外文)", None), 
                            dept=data.get("系所名稱(外文)", None), 
                            abstract=data.get("摘要(外文)", None),  
                            authors=data.get("作者(外文)", "").splitlines(),
                            advisors=data.get("指導教授(外文)", "").splitlines()
                        ), 
                        link=data.get("博碩士論文網址", None), 
                        keywords=[]
                    )
                    counter += 1

    def load(self) -> Iterator[List[Document]]:
        for doc in self.stream():
            yield from self._handle(doc)
        yield from self._flush()

    def _handle(self, document: Document) -> Iterator[List[Document]]:
        self.documents.append(document)
        if len(self.documents) >= self.buffer_size:
            yield from self._flush()

    def _flush(self) -> Iterator[List[Document]]:
        if self.documents:
            yield self.documents
            self.documents = []



class LitSearchDataLoader(DataLoader):
    def __init__(self, buffer_size: int = 64):
        """
        Initializes the LitSearchDataLoader.

        Args:
            buffer_size (int): Number of documents per batch. Defaults to 64.
        """
        self.buffer_size = buffer_size
        self.documents: List[Document] = []
        self.dataset = load_dataset("princeton-nlp/LitSearch", "corpus_s2orc", split="full")

    def stream(self) -> Iterator[Document]:
        for i, entry in enumerate(self.dataset):
            content = entry.get("content", {})
            annotations = content.get("annotations", {})

            text = content.get("text", "")
            title = self._extract_span(text, annotations, "title")
            abstract = self._extract_span(text, annotations, "abstract")
            authors = self._extract_list_of_spans(text, annotations, "author")

            #print(entry)
            try: 
                doc = LitSearchDocument(
                    corpusid=entry.get("corpusid", i),
                    externalids=coalesce(entry.get("externalids"), {}),
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    venue=content.get("venue"),
                    year=content.get("year"),
                    pdfurl=entry.get("source", {}).get("pdfurls", [None])[0],
                    text=text
                )
            except Exception as e:
                print(entry)
                print(f"Error processing entry {i}: {e}")
                raise e 
            
            yield doc

    def _extract_span(self, text: str, annotations: Dict[str, str], key: str) -> str:
        """Extract the first span of annotated text."""
        try:
            spans = json.loads(annotations.get(key, "[]"))
            if spans:
                span = spans[0]
                return text[span["start"]:span["end"]]
        except Exception:
            pass
        return ""

    def _extract_list_of_spans(self, text: str, annotations: Dict[str, str], key: str) -> List[str]:
        """_extract all spans of annotated text."""
        try:
            spans = json.loads(annotations.get(key, "[]"))
            return [text[span["start"]:span["end"]] for span in spans]
        except Exception:
            return []

    def load(self) -> Iterator[List[Document]]:
        for doc in self.stream():
            yield from self._handle(doc)
        yield from self._flush()

    def _handle(self, document: Document) -> Iterator[List[Document]]:
        self.documents.append(document)
        if len(self.documents) >= self.buffer_size:
            yield from self._flush()

    def _flush(self) -> Iterator[List[Document]]:
        if self.documents:
            yield self.documents
            self.documents = []

class Sampler: 
    def __init__(self, dataloader: DataLoader, max_samples: int = 1000):
        """
        Initializes the Sampler with a DataLoader.
        """
        self.dataloader = dataloader
        self.max_samples = max_samples

    def sample(self) -> Iterator[Document]: 
        """
        Samples a list of documents from a dataset
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
class PrefixSampler(Sampler):
    def sample(self) -> Iterator[Document]: 
        for i, doc in enumerate(self.dataloader.stream()):
            if i >= self.max_samples:
                break
            yield doc