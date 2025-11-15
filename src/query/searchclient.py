from typing import List, Dict, Any
import requests
from PIL import Image

from models.text_embedding import TextEmbedding
from models.image_embedding import ImageEmbedder
from models.reranker import CrossEncoderReranker


class SearchClient:
    def __init__(
        self,
        opensearch_url: str = "http://localhost:9200",
        text_index: str = "doc_chunks",
        image_index: str = "doc_images",
        text_embedder: TextEmbedding | None = None,
        image_embedder: ImageEmbedder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.opensearch_url = opensearch_url.rstrip("/")
        self.text_index = text_index
        self.image_index = image_index
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.reranker = reranker

    def knn_search(self,index: str, field: str, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        body = {
            "size": k,
            "query": {
                "knn": {
                    field: {  
                        "vector": query_vector,
                        "k": k,
                    }
                }
            },
        }

        resp = requests.post(
            f"{self.opensearch_url}/{index}/_search",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            print("OpenSearch error:", resp.text)
            raise
        data = resp.json()
        return data["hits"]["hits"]


    def search_text(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        vec = self.text_embedder.embed([query])[0]
        return self.knn_search(
            index=self.text_index,
            field="vector",
            query_vector=vec,
            k=k,
        )
    

    def search_text_reranked(self, query: str, knn_k: int = 50, top_k: int = 10) -> List[Dict[str, Any]]:
        hits = self.knn_search(
            index=self.text_index,
            field="vector",
            query_vector=self.text_embedder.embed([query])[0],
            k=knn_k,
        )

        docs = [h["_source"]["text"] for h in hits]
        scores = self.reranker.score(query, docs)

        for h, s in zip(hits, scores):
            h["_rerank_score"] = s

        hits = sorted(hits, key=lambda x: x["_rerank_score"], reverse=True)
        return hits[:top_k]


    def search_images(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        vec = self.image_embedder.embed_texts([query])[0]
        return self.knn_search(
            index=self.image_index,
            field="vector",
            query_vector=vec,
            k=k,
        )


    def search_multimodal(
        self,
        query: str,
        text_k: int = 5,
        image_k: int = 2,
        text_num_candidates: int = 50,
        image_num_candidates: int = 10,
        rerank_text: bool = True,
        text_knn_k: int = 50,
    ) -> Dict[str, List[Dict[str, Any]]]:

        if rerank_text and self.reranker is not None:
            text_hits = self.search_text_reranked(
                query=query,
                knn_k=text_knn_k,
                top_k=text_k,
                num_candidates=text_num_candidates,
            )
        else:
            text_hits = self.search_text(
                query=query,
                k=text_k,
                num_candidates=text_num_candidates,
            )

        image_hits = self.search_images(
            query=query,
            k=image_k,
            num_candidates=image_num_candidates,
        )

        return {
            "text_hits": text_hits,
            "image_hits": image_hits,
        }