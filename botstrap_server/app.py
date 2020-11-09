import json
import utils
import tensorflow as tf
import tensorflow_hub as hub

from fastapi import Body, FastAPI
from utils import generate_groups

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embedding_fn = hub.load(module_url)

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/label")
def label(utterances: list = Body(...), metric: str = 'euclidean'):
    embeddings = embedding_fn(utterances)
    groups = generate_groups(utterances, embeddings, metric)
    return groups
