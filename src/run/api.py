from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from src.run.app_state import AppState
from src.core.schema import AppConfig
from contextlib import asynccontextmanager

app = FastAPI()
app_state = AppState()

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
    app_state.load_all_metadata()
    yield
    # Shutdown logic (if needed)

app = FastAPI(lifespan=lifespan)

@app.get("/app_state/apps")
def list_apps():
    return app_state.list_apps()

@app.get("/app_state/app/{name}")
def get_app_metadata(name: str):
    try:
        return app_state.get_metadata(name)
    except KeyError:
        raise HTTPException(status_code=404, detail="App not found")

@app.post("/app_state/app")
def create_app(metadata: AppConfig):
    try:
        app_state.create_app(metadata)
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