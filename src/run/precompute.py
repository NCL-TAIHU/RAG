from src.core.vector_store import FileBackedDenseVS, VSMetadata, BaseVS
from src.core.embedder import BGEM3Embedder, DenseEmbedder, BaseEmbedder
from src.core.data import DataLoader
import yaml
import os 

DATASET = "ncl"
MODEL = "BAAI/bge-m3"
data_config = yaml.safe_load(open("config/data.yml", "r", encoding="utf-8"))
ROOT = os.path.join(data_config["root"]["path"], "vectors")
config = yaml.safe_load(open("config/model.yml", "r", encoding="utf-8"))

if __name__ == "__main__":
    dataloader = DataLoader.from_default(DATASET)
    embedder = BaseEmbedder.from_default(MODEL)
    vs = BaseVS.from_default(config[MODEL]["type"], os.path.join(ROOT, DATASET, MODEL))
    for docs in dataloader.load(): 
        contents = [next(iter(doc.content().values())).contents[0] for doc in docs]
        emb = embedder.embed(contents)
        ids = [doc.key() for doc in docs]
        vs.insert(ids, emb)
    vs.save()