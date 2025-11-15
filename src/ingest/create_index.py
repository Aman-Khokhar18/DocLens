import requests
from models.text_embedding import TextEmbedding  
from ingest.chunker import Chunk

OPENSEARCH_URL = "http://localhost:9200"
DOC_CHUNKS_INDEX = "doc_chunks"
DOC_IMAGE_INDEX = "doc_images"


def get_embedding_dim(embedder: TextEmbedding) -> int:
    vec = embedder.embed(["__dim_probe__"])[0]
    return len(vec)


def create_index(embedder: TextEmbedding) -> None:
    dim = get_embedding_dim(embedder)

    mapping = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "chunk_id":     {"type": "keyword"},
                "doc_id":       {"type": "keyword"},
                "source_file":  {"type": "keyword"},
                "section_path": {"type": "keyword"},
                "text":         {"type": "text"},
                "entity_ids":   {"type": "keyword"},
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

    requests.delete(f"{OPENSEARCH_URL}/{DOC_CHUNKS_INDEX}")

    resp = requests.put(
        f"{OPENSEARCH_URL}/{DOC_CHUNKS_INDEX}",
        json=mapping,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()


def index_chunk(chunk: Chunk, entity_ids, embedder: TextEmbedding) -> None:
    vec = embedder.embed([chunk.text])[0]
    doc = {
        "chunk_id":     chunk.chunk_id,
        "doc_id":       chunk.doc_id,
        "source_file":  chunk.source_file,
        "section_path": chunk.section_path,
        "text":         chunk.text,
        "entity_ids":   entity_ids,
        "vector":       vec,
    }

    resp = requests.post(
        f"{OPENSEARCH_URL}/{DOC_CHUNKS_INDEX}/_doc/{chunk.chunk_id}",
        json=doc,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
