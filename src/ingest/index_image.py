import requests
from PIL import Image
from typing import List

from models.image_embedding import ImageEmbedder

OPENSEARCH_URL = "http://localhost:9200"
DOC_IMAGES_INDEX = "doc_images"

def get_image_embedding_dim(embedder: ImageEmbedder) -> int:
    dummy = Image.new("RGB", (32, 32), color=0)
    vec = embedder.embed_images([dummy])[0]
    return len(vec)


def create_image_index(embedder: ImageEmbedder) -> None:
    dim = get_image_embedding_dim(embedder)

    mapping = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "image_id": {"type": "keyword"},
                "doc_id":   {"type": "keyword"},   # derived from path / parent folder
                "path":     {"type": "keyword"},   # path to file on disk
                "vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            }
        },
    }

    requests.delete(f"{OPENSEARCH_URL}/{DOC_IMAGES_INDEX}")

    resp = requests.put(
        f"{OPENSEARCH_URL}/{DOC_IMAGES_INDEX}",
        json=mapping,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()


def index_image(image_id: str, path: str, vector: List[float], doc_id: str = ""):
    doc = {
        "image_id": image_id,
        "path": path,
        "vector": vector,
    }

    resp = requests.post(
        f"{OPENSEARCH_URL}/{DOC_IMAGES_INDEX}/_doc/{image_id}",
        json=doc,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
