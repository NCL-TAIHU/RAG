import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from typing import Union, List, Dict
import logging
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
from milvus_model.hybrid import BGEM3EmbeddingFunction
from scipy.sparse import csr_array, coo_array
logger = logging.getLogger(__name__)

class DenseEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts into a list of vectors.
        :param texts: A single text or a list of texts to be embedded.
        :return: A list of vectors representing the embedded texts.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class SparseEmbedder:
    def embed(self, texts: List[str]) -> csr_array:
        """
        Embeds a list of texts into a list of sparse vectors.
        :param texts: A single text or a list of texts to be embedded.
        :return: A list of dictionaries representing the sparse vectors of the embedded texts.
        A dictionary should have the following structure:
        {
            "indices": List[int],  # indices of the non-zero elements in the sparse vector
            "values": List[float],  # learned importance
            "dim": 384000 # total dimension of the sparse vector
        }
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
class AutoModelEmbedder(DenseEmbedder):
    '''
    huggingface embedder, uses a pretrained model to embed texts. Produces dense vectors.
    '''
    def __init__(self, model_name: str):
        """
        Initializes the Embedder with a model.
        param model: The model to be used for embedding.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = 32  # Default batch size for embedding

    def embed(self, texts: List[str]) -> List[List[float]]:
        '''
        Embeds a list of texts into a list of vectors.
        '''
        if texts is None or len(texts) == 0:
            logger.warning("No texts provided for embedding.")
            return []
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        return embeddings

    def _embed_batch(self, batch_texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state #[batch_size, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"].unsqueeze(-1) # shape: [batch_size, seq_len, 1]
            count = attention_mask.sum(dim=1).clamp(min=1e-9) # shape: [batch_size, 1]
            summed = (hidden_states * attention_mask).sum(dim=1)
            mean_pooled = summed / count
            return mean_pooled.cpu().tolist() # shape: (len(texts), hidden_dim)
        

class BGEM3Embedder(SparseEmbedder):
    """
    BGEM3Embedder uses the BGEM3EmbeddingFunction to embed texts into sparse vectors.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16=True):
        """
        Initializes the BGEM3Embedder with a model.
        :param model_name: The name of the model to be used for embedding.
        """
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.model.to(self.device)  # Make sure the internal model is moved to GPU
        print(f"Model is on device: {self.device}")


    def embed(self, texts: List[str]) -> List[Dict]:
        output = self.model.encode(
            texts,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False, 
            batch_size=32,  # Default batch size for embedding
        )
        results = []
        for lw in output['lexical_weights']:
            sparse_embeding = {}
            sparse_embeding["indices"] = [int(idx) for idx in lw.keys()]
            sparse_embeding["values"] = [float(val) for val in lw.values()]
            sparse_embeding["dim"] = self.model.tokenizer.vocab_size
            results.append(sparse_embeding)
        return results
    
class MilvusBGEM3Embedder(SparseEmbedder):
    """
    MilvusBGEM3Embedder uses the BGEM3EmbeddingFunction to embed texts into sparse vectors.
    This class is specifically designed to work with Milvus for storing and retrieving embeddings.
    """
    def __init__(self):
        """
        Initializes the MilvusBGEM3Embedder with a model.
        :param model_name: The name of the model to be used for embedding.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.func = BGEM3EmbeddingFunction(device=self.device)

    def embed(self, texts: List[str]) -> csr_array:
        outputs = self.func(texts)
        return outputs["sparse"]