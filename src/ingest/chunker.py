import re, hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Iterable, Optional, Tuple

H_RE       = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
TITLE_RE   = re.compile(r"^#\s+(.*)$", re.MULTILINE)
SOURCE_RE  = re.compile(r"^>\s*Source:\s*(\S+)", re.IGNORECASE | re.MULTILINE)

def _hash_id(*parts: str, n: int = 16) -> str:
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:n]

def _title_url(md: str) -> Tuple[Optional[str], Optional[str]]:
    t = TITLE_RE.search(md)
    s = SOURCE_RE.search(md)
    return (t.group(1).strip() if t else None, s.group(1).strip() if s else None)

def _split_sections(md: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    lines = md.splitlines()
    path: List[str] = []
    buf: List[str] = []

    def flush():
        if buf:
            sections.append(
                (
                    " > ".join([p for p in path if p]) or "(root)",
                    "\n".join(buf).strip() + "\n",
                )
            )

    for ln in lines:
        m = H_RE.match(ln)
        if m:
            # new heading → finish previous section
            flush()
            level = len(m.group(1))
            text  = m.group(2).strip()
            path  = path[:level-1] + [text]
            buf   = [ln]  # include heading itself
        else:
            buf.append(ln)
    flush()
    return sections

def _char_windows(s: str, size: int, overlap: int) -> List[Tuple[int, int]]:
    assert size > 0 and 0 <= overlap < size
    out: List[Tuple[int, int]] = []
    i, n = 0, len(s)
    while i < n:
        j = min(i + size, n)
        out.append((i, j))
        if j >= n:
            break
        i = max(j - overlap, i + 1)
    return out

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    source_file: str
    section_path: str


class SectionChunker:

    def __init__(self, chunk_chars: int = 800, overlap_chars: int = 120, max_header_level: int = 6):
        assert 1 <= max_header_level <= 6
        assert 0 <= overlap_chars < chunk_chars
        self.size = int(chunk_chars)
        self.over = int(overlap_chars)
        self.max_h = max_header_level

    def chunk_markdown_text(self, md_text: str, source_file: str) -> List[Chunk]:
        # we still compute doc_id from URL or source_file for grouping
        _, url = _title_url(md_text)
        doc_key = url or source_file
        doc_id  = _hash_id(doc_key)

        sections = _split_sections(md_text)

        chunks: List[Chunk] = []
        idx = 0
        for sec_path, sec_text in sections:
            for (ws, we) in _char_windows(sec_text, self.size, self.over):
                piece = sec_text[ws:we].strip()
                if not piece:
                    continue
                chunk_id = _hash_id(doc_id, sec_path, str(idx))
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=piece,
                        source_file=source_file,
                        section_path=sec_path,
                    )
                )
                idx += 1
        return chunks

    def chunk_dir(self, root: Path) -> Iterable[Chunk]:
        root = Path(root)
        for p in sorted(root.rglob("*.md")):
            if "/assets/" in p.as_posix():
                continue
            md_text = p.read_text(encoding="utf-8", errors="ignore")
            rel = str(p.relative_to(root))
            for ch in self.chunk_markdown_text(md_text, rel):
                yield ch

    @staticmethod
    def write_jsonl(chunks: Iterable[Chunk], out_path: str | Path) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
