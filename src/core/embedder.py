import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from typing import Union, List, Dict, Any
import logging
from FlagEmbedding import BGEM3FlagModel
from milvus_model.hybrid import BGEM3EmbeddingFunction
from scipy.sparse import csr_array, coo_array, vstack
import numpy as np
import contextlib
import io
import GPUtil
import traceback
import yaml
from src.core.schema import EmbedderConfig, AutoModelEmbedderConfig, BGEEmbedderConfig


logger = logging.getLogger('taihu')
model_config = yaml.safe_load(open("config/model.yml", "r"))

class BaseEmbedder:
    """
    Base class for all emb
    """
    def embed(self, texts: List[str]) -> Any: 
        """
        Embeds a list of texts into a list of vectors.
        :param texts: A single text or a list of texts to be embedded.
        :return: A list of vectors representing the embedded texts.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def name(self) -> str: 
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def embed_chunks(self, ids: List[str], chunks_lst: List[List[str]]) -> Dict[str, Union[List[List[float]], csr_array]]:
        flat_chunks = [chunk for chunks in chunks_lst for chunk in chunks]
        embedded = self.embed(flat_chunks)
        pointer = 0
        embeddings = {}
        for doc_id, chunks in zip(ids, chunks_lst):
            n = len(chunks)
            embeddings[doc_id] = embedded[pointer: pointer + n]
            pointer += n
        return embeddings
    
    @classmethod
    def from_config(cls, config: EmbedderConfig) -> 'BaseEmbedder': 
        if config.type == "auto_model": 
            return AutoModelEmbedder.from_config(config)
        elif config.type == "bge":
            return BGEM3Embedder.from_config(config)
        else:
            raise ValueError(f"Unknown embedder type: {config.type}. Supported types: 'auto_model', 'bge'.")

    @classmethod
    def from_default(cls, model_name: str) -> 'BaseEmbedder':
        if model_config[model_name]["type"] == "dense":
            return DenseEmbedder.from_default(model_name)
        elif model_config[model_name]["type"] == "sparse":
            return SparseEmbedder.from_default(model_name)
        else:
            raise ValueError(f"Unknown embedder type: {model_config[model_name]['type']}. Supported types: 'dense', 'sparse'.")
    
class DenseEmbedder(BaseEmbedder):
    def get_dim(self) -> int:
        """
        Returns the dimension of the embedding vectors.
        :return: The dimension of the embedding vectors.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts into a list of vectors.
        :param texts: A single text or a list of texts to be embedded.
        :return: A list of vectors representing the embedded texts.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @classmethod
    def from_default(cls, model_name: str) -> 'DenseEmbedder':
        """
        Factory method to create a DenseEmbedder instance from a default model name.
        :param model_name: The name of the model to be used for embedding.
        :return: An instance of DenseEmbedder.
        """
        return AutoModelEmbedder(model_name)

class SparseEmbedder(BaseEmbedder):
    def embed(self, texts: List[str]) -> csr_array:
        """
        Embeds a list of texts into sparse vectors.
        :param texts: A single text or a list of texts to be embedded.
        :return: A sparse matrix representing the embedded texts.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @classmethod
    def from_default(cls, model_name: str) -> 'SparseEmbedder':
        """
        Factory method to create a SparseEmbedder instance from a default model name.
        :param model_name: The name of the model to be used for embedding.
        :return: An instance of SparseEmbedder.
        """
        return BGEM3Embedder(model_name)
    
class AutoModelEmbedder(DenseEmbedder):
    '''
    huggingface embedder, uses a pretrained model to embed texts. Produces dense vectors.
    '''
    def __init__(self, model_name: str):
        """
        Initializes the Embedder with a model.
        param model: The model to be used for embedding.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        try:
            # Relaxed GPU criteria - allow higher load and memory usage
            device_id = GPUtil.getFirstAvailable(order='memory', maxLoad=0.95, maxMemory=0.95)[0]
            self.device = torch.device(f"cuda:{device_id}")
        except Exception as e:
            logger.warning(f"No suitable GPU found: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()
        self.batch_size = 32  # Default batch size for embedding
        logger.info(f"Model {model_name} is on device: {self.device}")

    @classmethod
    def from_config(cls, config: AutoModelEmbedderConfig) -> 'AutoModelEmbedder':
        """
        Factory method to create an AutoModelEmbedder instance from a configuration.
        :param config: Configuration object containing model parameters.
        :return: An instance of AutoModelEmbedder.
        """
        return cls(config.model_name)

    def name(self) -> str:
        return self.model_name
    
    def get_dim(self) -> int:
        """
        Returns the dimension of the embedding vectors.
        :return: The dimension of the embedding vectors.
        """
        return self.model.config.hidden_size
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        '''
        Embeds a list of texts into a list of vectors.
        '''
        if texts is None or len(texts) == 0:
            #logger.warning("No texts provided for embedding." + "".join(traceback.format_stack()))
            logger.warning("No texts provided for embedding.")
            logger.debug("Stack trace:\n" + "".join(traceback.format_stack()))
            return []
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        return embeddings

    def _embed_batch(self, batch_texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            self.model.to(self.device)
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
        self.model_name = model_name
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        try:
            # Relaxed GPU criteria - allow higher load and memory usage
            device_id = GPUtil.getFirstAvailable(order='memory', maxLoad=0.95, maxMemory=0.95)[0]
            self.device = torch.device(f"cuda:{device_id}")
        except Exception as e:
            logger.warning(f"No suitable GPU found: {e}. Falling back to CPU.")
            self.device = torch.device("cpu")

        self.model.model.to(self.device)  # Make sure the internal model is moved to GPU
        self.vocab_size = len(self.model.tokenizer)
        logger.info(f"BGE Model is on device: {self.device}")

    @classmethod
    def from_config(cls, config: BGEEmbedderConfig) -> 'BGEM3Embedder':
        """
        Factory method to create a BGEM3Embedder instance from a configuration.
        :param config: Configuration object containing model parameters.
        :return: An instance of BGEM3Embedder.
        """
        return cls(config.model_name)
    
    def name(self) -> str:
        return self.model_name
    
    def embed(self, texts: List[str]) -> csr_array:
        f = io.StringIO()
        #redirect redundant output to avoid cluttering the console
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            output = self.model.encode_single_device(
                sentences=texts,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
                batch_size=32,
            )
            
        sparse_rows = []
        for lw in output["lexical_weights"]:
            indices = [int(k) for k in lw.keys()]
            values = np.array(list(lw.values()), dtype=np.float32)
            row_indices = np.zeros(len(indices), dtype=np.int32)
            row = csr_array((values, (row_indices, indices)), shape=(1, self.vocab_size))
            assert row.shape[0] == 1
            sparse_rows.append(row)

        return vstack(sparse_rows).tocsr()
    
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