from src.core.document import Document
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
from src.core.schema import RerankerConfig

class BaseReranker: 
    """
    Interface for reranking models.
    """
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank a list of documents based on the given query.

        Args:
            query (str): The search query.
            documents (list): A list of documents to rerank.

        Returns:
            List: A list of reranked documents.
        """
        raise NotImplementedError("Reranker is an abstract class and must be implemented by subclasses.")
    
    @classmethod
    def from_config(cls, config: RerankerConfig) -> 'BaseReranker':
        """
        Factory method to create a Reranker instance from a configuration.
        :param config: Configuration object containing reranker parameters.
        :return: An instance of Reranker.
        """
        if config.type == "identity":
            return IdentityReranker()
        elif config.type == "auto_model":
            return AutoModelReranker.from_config(config)

class IdentityReranker(BaseReranker):
    """
    A simple reranker that returns documents in the order they were provided.
    """
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        return documents

class AutoModelReranker(BaseReranker):
    """
    A reranker that uses a pretrained transformer-based cross-encoder model
    to score (query, document) pairs and sort the documents accordingly.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def _doc_to_string(self, doc: Document) -> str:
        """
        Flatten a document’s content fields into a single string.
        """
        content_fields = doc.content()
        return " ".join(
            f"{field.name}: {' '.join(map(str, field.contents))}" 
            for field in content_fields.values()
        )

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        # Construct query-document pairs
        pairs: List[Tuple[str, str]] = [(query, self._doc_to_string(doc)) for doc in documents]

        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        # Score with model
        with torch.no_grad():
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()

        # Zip with documents and sort by score
        reranked = sorted(zip(documents, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked]