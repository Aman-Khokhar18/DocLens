from dataclasses import dataclass
from typing import List, Optional
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

@dataclass
class ImageEmbedderConfig:
    model_id: str = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
    device: str = "cpu"
    batch_size: int = 16
    enable_text: bool = True   

class ImageEmbedder:

    def __init__(self, cfg: ImageEmbedderConfig = ImageEmbedderConfig()):
        self.cfg = cfg
        self.processor = CLIPProcessor.from_pretrained(cfg.model_id)
        self.model = CLIPModel.from_pretrained(cfg.model_id)
        self.model.to(cfg.device)
        self.model.eval()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.cfg.enable_text:
            raise RuntimeError("Text path disabled; set enable_text=True.")
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
            feats = self.model.get_text_features(**inputs)
            return feats.cpu().numpy().tolist()

    def embed_images(self, images: List["PIL.Image.Image"]) -> List[List[float]]:
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
            feats = self.model.get_image_features(**inputs)
            return feats.cpu().numpy().tolist()

    @staticmethod
    def cosine_sim_matrix(image_embs: List[List[float]], text_embs: List[List[float]]) -> List[List[float]]:
        def normalize(x):
            x = x.astype("float32")
            n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
            return x / n
        I = normalize(np.asarray(image_embs))
        T = normalize(np.asarray(text_embs))
        return (I @ T.T).tolist()
