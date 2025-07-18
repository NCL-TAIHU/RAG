from src.run.app import SearchApp
from src.core.data import DataLoader
from src.core.library import Library, InMemoryLibrary
from src.core.search_engine import HybridSearchEngine, MilvusSearchEngine, ElasticSearchEngine, SearchEngine, HybridMilvusSearchEngine
from src.core.document import Document
from src.core.filter import Filter
from src.core.embedder import SparseEmbedder, DenseEmbedder
from src.core.reranker import IdentityReranker
from src.core.manager import Manager
from src.core.router import Router
from src.core.reranker import Reranker
from src.core.vector_manager import VectorManager
from src.core.vector_store import SparseVS, DenseVS, BaseVS, FileBackedDenseVS, FileBackedSparseVS, VSMetadata
from src.core.chunker import BaseChunker, LengthChunker, SentenceChunker
from src.core.util import coalesce
import yaml
from typing import List
import os
from datetime import datetime

data_config = yaml.safe_load(open("config/data.yml", "r", encoding="utf-8"))
model_config = yaml.safe_load(open("config/model.yml", "r", encoding="utf-8"))
ROOT = os.path.join(data_config["root"]["path"])

class AppFactory:
    def __init__(self, 
                 engines: List[SearchEngine], 
                 library: Library, 
                 dataloader: DataLoader, 
                 router_name: str, #router is delayed initialization because it depends on the search engines
                 reranker: Reranker, 
                 max_files: int = 1000
        ):
        """
        Initializes the AppFactory with the provided search engines, library, and data loader.
        :param engines: List of search engines to be used in the application.
        :param library: The library where documents are stored.
        :param dataloader: The data loader for loading documents.
        :param max_files: Maximum number of files to process.
        """
        self.engines = engines
        self.library = library
        self.dataloader = dataloader
        self.router_name = router_name
        self.reranker = reranker    
        self.max_files = max_files

    @classmethod
    def from_default(cls, name: str, dataset: str) -> "AppFactory":
        '''
        Factory method to create an AppFactory instance based on the dataset name.
        :param name: The name of the default system configuration.
        :param dataset: The name of the dataset to use.
        '''
        if name == 'dev': 
            DOC_CLS = Document.from_dataset(dataset)  # Default document class based on dataset
            FILT_CLS = Filter.from_dataset(dataset)  # Default filter class based on dataset
            SPARSE_EMBEDDER_NAME = "BAAI/bge-m3"
            DENSE_EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
            CHANNEL = "abstract_chinese"
            sparse_embedder: SparseEmbedder = SparseEmbedder.from_default(SPARSE_EMBEDDER_NAME)
            dense_embedder: DenseEmbedder = DenseEmbedder.from_default(DENSE_EMBEDDER_NAME)
            now = datetime.now().isoformat()
            chunker = LengthChunker(chunk_length=512)
            chunker_type = chunker.metadata().chunker_type

            SPARSE_VS_ROOT = os.path.join(ROOT, dataset, SPARSE_EMBEDDER_NAME, CHANNEL, chunker_type)
            DENSE_VS_ROOT = os.path.join(ROOT, dataset, DENSE_EMBEDDER_NAME, CHANNEL, chunker_type) 
            sparse_vs: SparseVS = coalesce(lambda: FileBackedSparseVS.from_existing(SPARSE_VS_ROOT), #lambda is to delay evaluation
                                           lambda: FileBackedSparseVS(root=SPARSE_VS_ROOT, metadata=VSMetadata(
                                               embedding_type=model_config[SPARSE_EMBEDDER_NAME]["type"],
                                               dataset=dataset,
                                               channel=CHANNEL,
                                               chunker_meta=chunker.metadata(),
                                               model=model_config[SPARSE_EMBEDDER_NAME]["alias"],
                                               created_at=now,
                                               updated_at=now
                                           ))) 
            dense_vs: DenseVS = coalesce(lambda: FileBackedDenseVS.from_existing(DENSE_VS_ROOT),
                                        lambda: FileBackedDenseVS(root=DENSE_VS_ROOT, metadata=VSMetadata(
                                            embedding_type=model_config[DENSE_EMBEDDER_NAME]["type"],
                                            dataset=dataset,
                                            channel=CHANNEL,
                                            chunker_meta=chunker.metadata(),
                                            model=model_config[DENSE_EMBEDDER_NAME]["alias"],
                                            created_at=now,
                                            updated_at=now
                                        )))
            svm = VectorManager(
                vector_store=sparse_vs, 
                embedder=sparse_embedder, 
                chunker=chunker, 
                dataset=dataset,
                channel=CHANNEL
            )
            dvm = VectorManager(
                vector_store=dense_vs, 
                embedder=dense_embedder, 
                chunker=chunker, 
                dataset=dataset,
                channel=CHANNEL
            )
            sparse_milvus = MilvusSearchEngine(
                vector_type="sparse",
                vector_manager=svm,
                document_cls=DOC_CLS,
                filter_cls=FILT_CLS,
                force_rebuild=True
            )
            dense_milvus = MilvusSearchEngine(
                vector_type="dense",
                vector_manager=dvm,
                document_cls=DOC_CLS,
                filter_cls=FILT_CLS,
                force_rebuild=True
            )

            hybrid_milvus = HybridMilvusSearchEngine(
                document_cls=DOC_CLS,
                filter_cls=FILT_CLS,
                dense_vm=dvm,
                sparse_vm=svm,
                alpha=0.5,  # Default alpha value for hybrid search
                force_rebuild=True
            )

            elastic = ElasticSearchEngine(
                "https://localhost:9200", 
                document_cls=DOC_CLS, 
                filter_cls=FILT_CLS, 
                es_index=f"{dataset}"
            ) #relational search engine

            engine = HybridSearchEngine(
                relational_search_engine=elastic,
                vector_search_engine=hybrid_milvus
            )
            library: Library = InMemoryLibrary()
            dataloader = DataLoader.from_default(dataset)
            router = "simple"
            reranker = IdentityReranker()
            return cls(engines=[engine], library=library, dataloader=dataloader,
                       router_name=router, reranker=reranker, max_files=1000)


    def build(self) -> SearchApp: 
        manager = Manager(
            library=self.library,
            search_engines=self.engines,
            reranker=self.reranker,
            router_name=self.router_name
        )
        app = SearchApp(self.dataloader, manager, max_files=self.max_files)
        app.setup()
        return app