from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import main


class FakeEncoder:
    name = "fake"

    def encode(self, items):
        out = []
        for item in items:
            if isinstance(item, str):
                out.append([1.0, 0.0] if "red" in item else [0.0, 1.0])
            else:
                r, g, b = item.convert("RGB").getpixel((0, 0))
                out.append([r, b])
        return np.array(out, dtype=np.float32)


@pytest.fixture
def client(tmp_path):
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(red)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(blue)
    main.collections.clear()
    main.collections["flowers"] = main.Collection(
        "flowers", "image", FakeEncoder(), np.array([[1.0, 0.0], [0.0, 1.0]]),
        [{"label": "red flower", "path": str(red)}, {"label": "blue flower", "path": str(blue)}])
    main.collections["news"] = main.Collection(
        "news", "text", FakeEncoder(), np.array([[0.0, 1.0], [1.0, 0.0]]),
        [{"label": "sci.space", "text": "rockets"}, {"label": "rec.autos", "text": "my red car"}])
    yield TestClient(main.app)
    main.collections.clear()


def png_bytes(color):
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_collections(client):
    names = {c["name"] for c in client.get("/collections").json()}
    assert names == {"flowers", "news"}


def test_text_search_on_news(client):
    response = client.post("/search/text", json={"collection": "news", "text": "red", "k": 1})
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["label"] == "rec.autos"
    assert result["text"] == "my red car"


def test_text_search_on_images_returns_urls(client):
    response = client.post("/search/text", json={"collection": "flowers", "text": "a red one", "k": 2})
    results = response.json()["results"]
    assert results[0]["label"] == "red flower"
    assert results[0]["image_url"].endswith("/items/flowers/0/image")


def test_image_search(client):
    response = client.post("/search/image?collection=flowers&k=1",
                           files={"file": ("q.png", png_bytes((0, 0, 250)), "image/png")})
    assert response.status_code == 200
    assert response.json()["results"][0]["label"] == "blue flower"


def test_image_search_rejects_text_collection(client):
    response = client.post("/search/image?collection=news",
                           files={"file": ("q.png", png_bytes((0, 0, 250)), "image/png")})
    assert response.status_code == 400


def test_bad_file(client):
    response = client.post("/search/image", files={"file": ("q.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


def test_unknown_collection_and_bad_k(client):
    assert client.post("/search/text", json={"collection": "cars", "text": "x"}).status_code == 404
    assert client.post("/search/text", json={"collection": "news", "text": "x", "k": 0}).status_code == 422
    assert client.post("/search/text", json={"collection": "news", "text": ""}).status_code == 422


def test_item_image(client):
    assert client.get("/items/flowers/1/image").status_code == 200
    assert client.get("/items/flowers/9/image").status_code == 404
    assert client.get("/items/news/0/image").status_code == 404
