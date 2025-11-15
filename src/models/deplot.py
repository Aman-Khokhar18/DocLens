from dataclasses import dataclass
from typing import Optional
import torch
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image

@dataclass
class DePlotConfig:
    model_id: str = "google/deplot"
    device: str = "cpu"
    max_new_tokens: int = 256
    prompt: str = "Generate a table of values from this chart."

class DePlot:
    def __init__(self, cfg: DePlotConfig = DePlotConfig()):
        self.cfg = cfg
        self.processor = Pix2StructProcessor.from_pretrained(cfg.model_id)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(cfg.model_id)
        self.model.to(cfg.device)
        self.model.eval()

    def infer(self, image: "PIL.Image.Image", question: Optional[str] = None) -> str:
        question = question or self.cfg.prompt
        with torch.no_grad():
            inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.cfg.device)
            out = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens)
            return self.processor.decode(out[0], skip_special_tokens=True)
