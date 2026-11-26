import json
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from similarity.search import normalize, top_k

INDEX_DIR = Path("index")
collections = {}


class Collection:
    def __init__(self, name, kind, encoder, vectors, items):
        self.name = name
        self.kind = kind
        self.encoder = encoder
        self.vectors = normalize(vectors)
        self.items = items


def load_collections():
    from similarity.encoders import ClipEncoder, TextEncoder
    setup = {"flowers": ("image", ClipEncoder), "news": ("text", TextEncoder)}
    for name, (kind, make_encoder) in setup.items():
        vectors_file = INDEX_DIR / f"{name}.npy"
        items_file = INDEX_DIR / f"{name}.json"
        if not vectors_file.exists():
            print(f"skipping {name}: run python build_index.py first")
            continue
        items = json.loads(items_file.read_text())
        collections[name] = Collection(name, kind, make_encoder(), np.load(vectors_file), items)
        print(f"loaded {name}: {len(items)} items")


@asynccontextmanager
async def lifespan(app):
    if not collections:
        load_collections()
    yield


app = FastAPI(title="Similarity Is All You Need", lifespan=lifespan)


class TextQuery(BaseModel):
    collection: str
    text: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


def get_collection(name):
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"no collection called {name}")
    return collections[name]


def make_results(collection, hits, sims, request):
    results = []
    for i, score in zip(hits, sims):
        item = collection.items[int(i)]
        result = {"id": int(i), "label": item["label"], "score": round(float(score), 4)}
        if collection.kind == "image":
            result["image_url"] = str(request.url_for("item_image", collection=collection.name, item_id=int(i)))
        else:
            result["text"] = item["text"][:400]
        results.append(result)
    return {"collection": collection.name, "count": len(results), "results": results}


@app.get("/health")
def health():
    return {"status": "ok", "collections": list(collections)}


@app.get("/collections")
def list_collections():
    return [{"name": c.name, "kind": c.kind, "size": len(c.items), "encoder": c.encoder.name}
            for c in collections.values()]


@app.post("/search/text")
def search_text(query: TextQuery, request: Request):
    collection = get_collection(query.collection)
    vector = collection.encoder.encode([query.text])
    hits, sims = top_k(collection.vectors, vector, query.k)
    return make_results(collection, hits, sims, request)


@app.post("/search/image")
async def search_image(request: Request, collection: str = "flowers", k: int = 10, file: UploadFile = File(...)):
    found = get_collection(collection)
    if found.kind != "image":
        raise HTTPException(status_code=400, detail=f"{collection} only supports text search")
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="that file is not an image")
    k = max(1, min(k, 100))
    vector = found.encoder.encode([image])
    hits, sims = top_k(found.vectors, vector, k)
    return make_results(found, hits, sims, request)


@app.get("/items/{collection}/{item_id}/image", name="item_image")
def item_image(collection: str, item_id: int):
    found = get_collection(collection)
    if found.kind != "image" or not 0 <= item_id < len(found.items):
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(found.items[item_id]["path"])
