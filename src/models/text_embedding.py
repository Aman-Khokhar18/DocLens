from dataclasses import dataclass
from typing import List
from sentence_transformers import SentenceTransformer

@dataclass
class TextEmbedderConfig:
    model_id: str = "Salesforce/SFR-Embedding-Code-400M_R"  
    device: str = "cuda"
    normalize: bool = True
    trust_remote_code: bool = True
    batch_size: int = 32

class TextEmbedding:
    def __init__(self, cfg: TextEmbedderConfig = TextEmbedderConfig()):

        self.cfg = cfg
        self.model = SentenceTransformer(
            cfg.model_id, device=cfg.device, trust_remote_code=cfg.trust_remote_code
        )
        # warmup to build caches
        _ = self.model.encode(["warmup"], normalize_embeddings=cfg.normalize)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts, batch_size=self.cfg.batch_size, normalize_embeddings=self.cfg.normalize
        )
        return vecs.tolist()
