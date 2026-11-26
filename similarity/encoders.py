from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


class Encoder(Protocol):
    name: str

    def encode(self, items) -> np.ndarray: ...


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def open_image(item):
    if isinstance(item, Path):
        with Image.open(item) as img:
            return img.convert("RGB")
    return item.convert("RGB")


class ClipEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", batch_size=64):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.name = model_name.split("/")[-1]
        self.device = pick_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.batch_size = batch_size

    def encode(self, items):
        items = list(items)
        if len(items) == 0:
            return np.zeros((0, self.model.config.projection_dim), dtype=np.float32)
        if isinstance(items[0], str):
            return self.encode_texts(items)
        return self.encode_images(items)

    def encode_images(self, items):
        out = []
        with self.torch.no_grad():
            for start in range(0, len(items), self.batch_size):
                images = [open_image(item) for item in items[start:start + self.batch_size]]
                pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
                pooled = self.model.vision_model(pixel_values=pixels).pooler_output
                out.append(self.model.visual_projection(pooled).float().cpu().numpy())
        return np.concatenate(out)

    def encode_texts(self, texts):
        out = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = self.processor.tokenizer(texts[start:start + self.batch_size], padding=True,
                                                 truncation=True, max_length=77, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                pooled = self.model.text_model(input_ids=batch["input_ids"],
                                               attention_mask=batch["attention_mask"]).pooler_output
                out.append(self.model.text_projection(pooled).float().cpu().numpy())
        return np.concatenate(out)


class TextEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.name = model_name.split("/")[-1]
        self.model = SentenceTransformer(model_name, device=pick_device())
        self.batch_size = batch_size

    def encode(self, items):
        items = list(items)
        return self.model.encode(items, batch_size=self.batch_size, convert_to_numpy=True,
                                 show_progress_bar=len(items) > 1000)


class TfidfEncoder:
    def __init__(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.name = "tfidf"
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_features=50000)
        self.vectorizer.fit(texts)

    def encode(self, items):
        return self.vectorizer.transform(list(items)).astype(np.float32)


class ColorHistogramEncoder:
    def __init__(self, bins=8):
        self.name = "color-histogram"
        self.bins = bins

    def encode(self, items):
        out = []
        for item in items:
            img = open_image(item).resize((128, 128))
            pixels = np.asarray(img.convert("HSV")).reshape(-1, 3) // (256 // self.bins)
            codes = pixels[:, 0] * self.bins * self.bins + pixels[:, 1] * self.bins + pixels[:, 2]
            hist = np.bincount(codes, minlength=self.bins ** 3).astype(np.float32)
            out.append(np.sqrt(hist / hist.sum()))
        return np.array(out)
