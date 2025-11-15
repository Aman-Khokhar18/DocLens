from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from typing import Iterable, List, TypeVar, Dict, Any
from tqdm import tqdm
import hashlib
from PIL import Image

from models.text_embedding import TextEmbedding
from models.image_embedding import ImageEmbedder, ImageEmbedderConfig
from ingest.chunker import Chunk, SectionChunker
from ingest.extract_graph import extract_kg_from_chunk
from ingest.create_graph import driver, upsert_entities, upsert_relations
from ingest.create_index import create_index, index_chunk
from ingest.index_image import create_image_index, index_image


T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Iterable[List[T]]:
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

def _find_image_files(assets_root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return [
        p
        for p in assets_root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]


def ingest_docs(root_dir: str, enable_graph: bool = False, batch_size: int = 32):

    chunker = SectionChunker(chunk_chars=2500, overlap_chars=200)

    embedder = TextEmbedding()
    create_index(embedder)

    root_path = Path(root_dir)

    # Materialize chunks so tqdm can display total
    chunks: List[Chunk] = list(chunker.chunk_dir(root_path))
    total_chunks = len(chunks)

    if total_chunks == 0:
        print("No chunks found, nothing to ingest.")
        return

    if not enable_graph:
        print("Graph disabled → running plain vector RAG ingestion.")
        with tqdm(
            total=total_chunks,
            desc="Indexing chunks (no graph)",
            unit="chunk",
        ) as pbar:
            for chunk in chunks:
                entity_ids: List[str] = []
                index_chunk(chunk, entity_ids, embedder)
                pbar.update(1)
        return

    effective_batch_size = max(1, batch_size)
    print(
        f"Graph enabled → running Graph RAG ingestion "
        f"(batch_size={effective_batch_size})."
    )

    with driver.session() as session, ThreadPoolExecutor(
        max_workers=effective_batch_size
    ) as executor:
        with tqdm(
            total=total_chunks,
            desc="Processing chunks (graph + index)",
            unit="chunk",
        ) as pbar:
            for chunk_batch in batched(chunks, effective_batch_size):
                if not chunk_batch:
                    continue

                futures = {
                    executor.submit(extract_kg_from_chunk, chunk.text): idx
                    for idx, chunk in enumerate(chunk_batch)
                }

                kg_list: List[Dict[str, Any]] = [None] * len(chunk_batch)

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        kg_list[idx] = future.result()
                    except Exception:
                        kg_list[idx] = {"entities": [], "relations": []}

                for chunk, kg in zip(chunk_batch, kg_list):
                    if kg is None:
                        kg = {"entities": [], "relations": []}

                    entities = kg.get("entities", []) or []
                    relations = kg.get("relations", []) or []

                    if entities:
                        session.execute_write(
                            upsert_entities,
                            entities=entities,
                            source_file=chunk.source_file,
                            section_path=chunk.section_path,
                            doc_id=chunk.doc_id,
                        )

                    if relations:
                        session.execute_write(
                            upsert_relations,
                            relations=relations,
                            source_file=chunk.source_file,
                        )

                    entity_ids = [e.get("id") for e in entities if e.get("id")]
                    index_chunk(chunk, entity_ids, embedder)

                pbar.update(len(chunk_batch))


    # ----- Image Ingestion ---------------------------------------------------------------------
    assets_root = Path(root_dir) / "assets"
    if not assets_root.exists():
        print(f"No assets folder found at {assets_root}, skipping image ingestion.")
        return

    img_embedder = ImageEmbedder(ImageEmbedderConfig(device="cpu")) 
    create_image_index(img_embedder)

    image_paths = _find_image_files(assets_root)
    total_images = len(image_paths)

    if total_images == 0:
        print("No images found in /assets, skipping image ingestion.")
        return

    print(f"Found {total_images} images under {assets_root} → indexing with CLIP")

    with tqdm(total=total_images, desc="Indexing images", unit="image") as pbar:
        for path_batch in batched(image_paths, img_embedder.cfg.batch_size):
            pil_images = []
            valid_paths = []

            for p in path_batch:
                try:
                    img = Image.open(p).convert("RGB")
                    pil_images.append(img)
                    valid_paths.append(p)
                except Exception:
                    continue

            if not pil_images:
                pbar.update(len(path_batch))
                continue

            vecs = img_embedder.embed_images(pil_images)

            for p, vec in zip(valid_paths, vecs):
                # simple stable id based on path
                image_id = hashlib.md5(p.as_posix().encode("utf-8")).hexdigest()

                index_image(
                    image_id=image_id,
                    path=p.as_posix(),
                    vector=vec,
                )

            pbar.update(len(path_batch))


if __name__ == "__main__":
    ingest_docs("scrapped_data/docs.stripe.com")
