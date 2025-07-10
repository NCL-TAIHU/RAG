from src.core.vector_store import FileBackedDenseVS, VSMetadata, BaseVS
from src.core.embedder import BGEM3Embedder, DenseEmbedder, BaseEmbedder
from src.core.data import DataLoader
from src.core.util import get_first_content, coalesce
import yaml
import os 
from tqdm import tqdm

DATASET = "ncl"
MODEL = "BAAI/bge-m3"
data_config = yaml.safe_load(open("config/data.yml", "r", encoding="utf-8"))
ROOT = os.path.join(data_config["root"]["path"], "vectors")
config = yaml.safe_load(open("config/model.yml", "r", encoding="utf-8"))

if __name__ == "__main__":
    dataloader = DataLoader.from_default(DATASET)
    embedder = BaseEmbedder.from_default(MODEL)
    root = os.path.join(ROOT, DATASET, MODEL)
    vs: BaseVS = coalesce(
        BaseVS.from_existing(root), 
        BaseVS.create(
            type=config[MODEL]["type"], 
            root=root, 
            metadata=VSMetadata.from_model(MODEL)
        )
    )
    for docs in tqdm(dataloader.load()): 
        contents = [get_first_content(doc) for doc in docs]
        emb = embedder.embed(contents)
        ids = [doc.key() for doc in docs]
        vs.insert(ids, emb)
    vs.save()