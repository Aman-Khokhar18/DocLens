from typing import List, Dict, Any
import requests
from PIL import Image

from models.text_embedding import TextEmbedding
from models.image_embedding import ImageEmbedder
from models.reranker import CrossEncoderReranker
from query.searchclient import SearchClient

client = SearchClient(
    opensearch_url="http://localhost:9200",
    text_index="doc_chunks",
    image_index="doc_images",
    text_embedder=TextEmbedding(),
    image_embedder=ImageEmbedder(),
    reranker=CrossEncoderReranker()
)

query = "How do i make payments with Stripe?"

results = client.search_text(
    query=query, k=10
)

for i, hit in enumerate(results, start=1):
    src = hit["_source"]
    print(f"RESULT {i}")
    print("Score:", hit["_score"])
    print("Text:", src.get("text"))    
    print("-" * 60)
