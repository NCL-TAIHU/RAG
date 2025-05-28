from src.core.entity import Document, MetaData
import os

class DataLoader:
    def __init__(self, abstract_path, content_path, keywords_path, MAX_FILES=None):
        """
        Initializes the DataLoader with paths to the data files.
        
        :param abstract_path: Path to the abstract data file.
        :param content_path: Path to the content data file.
        :param keywords_path: Path to the keywords data file.
        """
        self.abstract_path = abstract_path
        self.content_path = content_path
        self.keywords_path = keywords_path
        self.documents = []
        self.metadatas = []
        self.MAX_FILES = MAX_FILES

    def load_data(self):
        document_paths = os.listdir(self.abstract_path)
        content_paths = os.listdir(self.content_path)
        keywords_paths = os.listdir(self.keywords_path)
        print(document_paths[:5])

        #use set intersection to find common files
        common_files = set(document_paths) & set(content_paths) & set(keywords_paths)
        print(f"Found {len(common_files)} common files in all directories.")

        for filename in common_files:
            with open(os.path.join(self.abstract_path, filename), 'r', encoding='utf-8') as f1, \
                    open(os.path.join(self.content_path, filename), 'r', encoding='utf-8') as f2, \
                    open(os.path.join(self.keywords_path, filename), 'r', encoding='utf-8') as f3:
                abstract = f1.read().strip()
                content = f2.read().strip()
                keywords = f3.read().strip().split(',')  # Assuming keywords are comma-separated
                doc_id = os.path.splitext(filename)[0]  # Use filename without extension as ID
                self.documents.append(Document(id=doc_id, abstract=abstract))
                self.metadatas.append(MetaData(id=doc_id, content=content, keywords=keywords))
          
        self.documents.sort(key=lambda x: x.id)
        self.metadatas.sort(key=lambda x: x.id)

    def get_documents(self) -> list[Document]:
        return self.documents[:self.MAX_FILES] if self.MAX_FILES else self.documents

    def get_metadatas(self) -> list[MetaData]:
        return self.metadatas[:self.MAX_FILES] if self.MAX_FILES else self.metadatas