from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from src.core.schema import AppConfig
from contextlib import asynccontextmanager
from adaptor import assemble_app_config 
from typing import Dict, Any, List
from src.run.state import BaseState
from src.core.vector_set import BaseVectorSet
from src.core.schema import VectorSetConfig, AppConfig
from src.core.app import SearchApp
import yaml
import os
app = FastAPI()
app_config = yaml.safe_load(open("config/app.yml", "r", encoding="utf-8"))
vector_set_config = yaml.safe_load(open("config/vector_set.yml", "r", encoding="utf-8"))


app_state = BaseState[AppConfig, SearchApp](
    config_cls=AppConfig,
    obj_cls=SearchApp,
    config_dir=app_config["config_path"]
)
vector_set_state = BaseState[VectorSetConfig, BaseVectorSet](
    config_cls=VectorSetConfig,
    obj_cls=BaseVectorSet,
    config_dir=vector_set_config["config_path"]
) 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    app_state.load_all_configs()
    vector_set_state.load_all_configs()
    yield
    # Shutdown logic (if needed)

app = FastAPI(lifespan=lifespan)

@app.get("/app_state/apps")
def list_apps():
    return app_state.list_apps()

@app.get("/app_state/app/{id}")
def get_app_metadata(id: str):
    try:
        return app_state.get_config(id)
    except KeyError:
        raise HTTPException(status_code=404, detail="App not found")

@app.post("/app_state/app")
def create_app(form: Dict[str, Any]):
    try:
        config = assemble_app_config(form)
        app_state.register_app(config)
        return {"status": "created"}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/app_state/activate/{name}")
def activate_app(name: str):
    try:
        app_state.activate_app(name)
        return {"status": "activated"}
    except KeyError:
        raise HTTPException(status_code=404, detail="App not found")

@app.delete("/app_state/app/{name}")
def delete_app(name: str):
    try:
        app_state.remove_app(name)
        return {"status": "removed"}
    except KeyError:
        raise HTTPException(status_code=404, detail="App not found")