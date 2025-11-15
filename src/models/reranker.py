from dataclasses import dataclass
from typing import List
from sentence_transformers import CrossEncoder

@dataclass
class CrossEncoderRerankerConfig:
    model_id: str = "BAAI/bge-reranker-base" 
    device: str = "cpu"
    max_length: int = 512
    batch_size: int = 16

class CrossEncoderReranker:
    
    def __init__(self, cfg: CrossEncoderRerankerConfig = CrossEncoderRerankerConfig()):
        self.cfg = cfg
        self.model = CrossEncoder(cfg.model_id, max_length=cfg.max_length, device=cfg.device)
        # warmup
        _ = self.model.predict([("warmup", "warmup")])

    def score(self, query: str, docs: List[str]) -> List[float]:
        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs, batch_size=self.cfg.batch_size)
        return [float(s) for s in scores]
