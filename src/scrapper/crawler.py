# src/scrapper/scrapper.py
import os, re, hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse

import scrapy
from scrapy.http import HtmlResponse
from bs4 import BeautifulSoup, NavigableString, Tag

try:
    from scrapy_playwright.page import PageMethod
except Exception:
    PageMethod = None


class DocsCrawler(scrapy.Spider):
    name = "minimal_docs_crawler"

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DEPTH_LIMIT": 2,
        "FEED_EXPORT_ENCODING": "utf-8",
        "LOG_LEVEL": "INFO",
        "DOWNLOAD_DELAY": 0.2,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.5,
        "AUTOTHROTTLE_MAX_DELAY": 3.0,
        "USER_AGENT": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"),

        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": 30000,
    }

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")

    def __init__(self, start_url=None, depth=None, **kwargs):
        super().__init__(**kwargs)
        if not start_url:
            raise ValueError("Provide -a start_url=https://...")
        self.start_urls = [start_url.strip()]
        if depth is not None:
            try:
                self.custom_settings["DEPTH_LIMIT"] = int(depth)
            except Exception:
                self.logger.warning("Invalid depth; using default DEPTH_LIMIT")
        self.allowed_domain = urlparse(self.start_urls[0]).netloc.split(":")[0]

        host = urlparse(start_url).netloc
        self.out_dir = Path(f"scrapped_data/{host}")
        self.assets_dir = self.out_dir / "assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        yield scrapy.Request(self.start_urls[0], callback=self.parse)

    def parse(self, response: HtmlResponse):
        # ---- Playwright fallback: 403 or effectively empty ----
        if (response.status == 403 or self._looks_empty(response)) and not response.meta.get("from_playwright"):
            self.logger.warning(f"{'403' if response.status == 403 else 'Empty'} at {response.url}; retrying with Playwright…")
            meta = {"playwright": True, "from_playwright": True}
            if PageMethod is not None:
                meta["playwright_page_methods"] = [
                    PageMethod("wait_for_load_state", state="domcontentloaded"),
                    PageMethod("wait_for_load_state", state="networkidle"),
                    PageMethod("wait_for_timeout", 400),
                ]
            yield scrapy.Request(response.url, callback=self.parse, dont_filter=True, meta=meta)
            return

        if (response.status == 403 or self._looks_empty(response)) and response.meta.get("from_playwright"):
            self.logger.warning(f"Skipping blocked/empty even with Playwright: {response.url}")
            return

        if not isinstance(response, HtmlResponse):
            return

        url = response.url
        soup = BeautifulSoup(response.text, "html.parser")

        canonical = soup.find("link", rel="canonical")
        canon_url = urljoin(url, canonical["href"].strip()) if canonical and canonical.get("href") else url

        main = (soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"})
                or soup.select_one("[data-docs-content], .markdown, .docContent, .prose") or soup.body)

        for sel in ["nav","header","footer","aside",".breadcrumbs",".toc","[role='navigation']",
                    ".cookie",".cookie-banner",".ads",".sidebar"]:
            for n in main.select(sel):
                n.decompose()

        title = (soup.title.string.strip() if soup.title and soup.title.string else "").strip()
        h1 = main.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)

        lines = []
        lines.append(f"# {title}" if title else "# (untitled)")
        lines.append("")
        lines.append(f"> Source: {canon_url}")
        lines.append("")

        # ---- Build Markdown ----
        for node in self._iter_blocks(main):
            name = node.name.lower()
            if name == "img":
                src = (node.get("src") or "").strip()
                if src:
                    abs_url = urljoin(canon_url, src)
                    fname = self._planned_asset_name(abs_url)
                    alt = (node.get("alt") or "").strip()
                    # schedule image download
                    yield scrapy.Request(abs_url, callback=self._save_image_response, dont_filter=True,
                                         meta={"planned_name": fname})
                    # add markdown reference
                    lines.append(f"![{alt}]({self._rel_asset(fname)})")
                    lines.append("")
                continue  # handled
            # Everything else goes through the helper (returns ONLY strings/lists)
            md = self._block_to_md(node, canon_url)
            if md:
                if isinstance(md, list):
                    lines.extend(md)
                else:
                    lines.append(md)

        # ---- Write .md ----
        md_name = self._page_slug(canon_url) + ".md"
        md_path = self.out_dir / md_name
        md_path.write_text("\n".join(self._squash_blank_lines(lines)).rstrip() + "\n", encoding="utf-8")
        self.logger.info(f"Saved markdown: {md_name}")

        # ---- Follow internal links ----
        for a in soup.find_all("a", href=True):
            href = (a["href"] or "").strip()
            if not href or href.startswith("#") or href.lower().startswith(("mailto:", "javascript:")):
                continue
            abs_url = urljoin(url, href)
            netloc = urlparse(abs_url).netloc.split(":")[0]
            if netloc != self.allowed_domain:
                continue
            if any(abs_url.lower().endswith(ext) for ext in self.IMG_EXTS):
                yield scrapy.Request(abs_url, callback=self._save_image_response, dont_filter=True)
            else:
                yield scrapy.Request(abs_url, callback=self.parse)

    # --------- helpers ---------
    def _looks_empty(self, response: HtmlResponse) -> bool:
        txt = (response.text or "").strip()
        if len(txt) < 800:
            return True
        soup = BeautifulSoup(response.text, "html.parser")
        return not bool(soup.find(["h1", "h2", "p", "pre", "article", "main"]))

    def _iter_blocks(self, root: Tag):
        for node in root.descendants:
            if isinstance(node, Tag):
                n = node.name.lower()
                if n in {"h1","h2","h3","h4","h5","h6","p","pre","ul","ol","blockquote","table","img"}:
                    yield node

    def _block_to_md(self, node: Tag, base_url: str):
        name = node.name.lower()
        if name.startswith("h") and len(name) == 2 and name[1].isdigit():
            lvl = int(name[1]); text = self._inline_text(node, base_url)
            return "#" * lvl + f" {text}"
        if name == "p":
            txt = self._inline_text(node, base_url)
            return [txt, ""] if txt else None
        if name == "pre":
            code = node.get_text().rstrip("\n")
            return ["```", code, "```", ""] if code else None
        if name in {"ul","ol"}:
            ordered = name == "ol"; lines = []
            self._render_list(node, lines, depth=0, ordered=ordered, base_url=base_url)
            lines.append(""); return lines
        if name == "blockquote":
            text = self._inline_text(node, base_url)
            return (["> " + line if line else ">"] + [""]) if text else None
        if name == "table":
            html = node.decode()
            return ["```html", html, "```", ""]
        # NOTE: images handled in parse(); don't handle here
        return None

    def _inline_text(self, tag: Tag, base_url: str) -> str:
        parts = []
        for el in tag.descendants:
            if isinstance(el, NavigableString):
                parts.append(str(el))
            elif isinstance(el, Tag):
                nm = el.name.lower()
                if nm == "code" and (el.parent.name.lower() != "pre"):
                    parts.append(f"`{el.get_text()}`")
                elif nm == "a" and el.get("href"):
                    href = urljoin(base_url, el.get("href").strip())
                    label = el.get_text(strip=True) or href
                    parts.append(f"[{label}]({href})")
        return " ".join("".join(parts).strip().split())

    def _render_list(self, ul_or_ol: Tag, lines: list, depth: int, ordered: bool, base_url: str):
        i = 1
        for li in ul_or_ol.find_all("li", recursive=False):
            prefix = f"{i}." if ordered else "-"
            head, tails = [], []
            for c in li.contents:
                if isinstance(c, NavigableString):
                    head.append(str(c))
                elif isinstance(c, Tag) and c.name.lower() not in {"ul","ol","pre","p"}:
                    head.append(self._inline_text(c, base_url))
                else:
                    tails.append(c)
            first_line = " ".join(" ".join(head).split()).strip()
            lines.append(("  " * depth) + f"{prefix} {first_line}".rstrip())
            for t in tails:
                if isinstance(t, Tag) and t.name.lower() in {"ul","ol"}:
                    self._render_list(t, lines, depth+1, ordered=(t.name.lower()=="ol"), base_url=base_url)
                elif isinstance(t, Tag) and t.name.lower() in {"pre","p"}:
                    if t.name.lower() == "p":
                        txt = self._inline_text(t, base_url)
                        if txt:
                            lines.append(("  " * (depth+1)) + txt)
                    else:
                        code = t.get_text().rstrip("\n")
                        if code:
                            indent = "  " * (depth+1)
                            lines.extend([indent + "```", code, indent + "```"])
            i += 1

    def _save_image_response(self, response):
        planned = response.meta.get("planned_name")
        if planned:
            fname = planned
        else:
            parsed = urlparse(response.url)
            name = os.path.basename(parsed.path) or "image"
            if "." not in name:
                ct = response.headers.get(b"Content-Type", b"").decode().lower()
                name += self._ext_from_ct(ct)
            root, ext = os.path.splitext(name)
            digest = hashlib.sha1(response.body[:64]).hexdigest()[:8]
            fname = f"{self._safe_name(root)}-{digest}{ext or '.bin'}"
        (self.assets_dir / fname).write_bytes(response.body)
        self.logger.info(f"Saved asset: {fname}")

    def _planned_asset_name(self, abs_url: str) -> str:
        parsed = urlparse(abs_url)
        base = os.path.basename(parsed.path) or "asset"
        root, ext = os.path.splitext(base)
        if not ext:
            ext = ""
        h = hashlib.sha1(abs_url.encode()).hexdigest()[:8]
        return f"{self._safe_name(root)}-{h}{ext or ''}"

    def _page_slug(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        tail = path.split("/")[-1] or "index"
        return self._safe_name(tail)

    @staticmethod
    def _rel_asset(name: str) -> str:
        return f"assets/{name}"

    @staticmethod
    def _ext_from_ct(ct: str) -> str:
        if "png" in ct: return ".png"
        if "jpeg" in ct or "jpg" in ct: return ".jpg"
        if "gif" in ct: return ".gif"
        if "webp" in ct: return ".webp"
        return ".bin"

    @staticmethod
    def _squash_blank_lines(lines):
        out, prev_blank = [], False
        for ln in lines:
            s = "" if ln is None else str(ln)
            blank = (s.strip() == "")
            if blank and prev_blank:
                continue
            out.append(s)
            prev_blank = blank
        return out

    @staticmethod
    def _safe_name(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "-", s)[:80] or "file"
