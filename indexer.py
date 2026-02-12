"""
Build-time indexer: downloads Databricks docs, chunks them, and builds a ChromaDB index.

Run during Docker build:
    python indexer.py

Sources:
    1. Databricks Terraform provider docs (github.com/databricks/terraform-provider-databricks)
    2. Databricks Python SDK docs (github.com/databricks/databricks-sdk-py)
    3. Databricks GenAI Cookbook (github.com/databricks/genai-cookbook)
    4. Databricks CLI docs (github.com/databricks/cli)
    5. Databricks public docs (docs.databricks.com) — scraped at build time
"""

import os
import re
import sys
import json
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.setrecursionlimit(10000)

import xml.etree.ElementTree as ET

import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from markdownify import markdownify as md

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPOS = [
    {
        "name": "terraform-provider-databricks",
        "url": "https://github.com/databricks/terraform-provider-databricks.git",
        "branch": "main",
        "glob": "docs/**/*.md",
        "collection": "terraform_databricks",
        "sparse": ["docs"],
    },
    {
        "name": "databricks-sdk-py",
        "url": "https://github.com/databricks/databricks-sdk-py.git",
        "branch": "main",
        "glob": "**/*.md",
        "collection": "databricks_api",
        "sparse": None,
    },
    {
        "name": "genai-cookbook",
        "url": "https://github.com/databricks/genai-cookbook.git",
        "branch": "main",
        "glob": "**/*.md",
        "collection": "databricks_genai",
        "sparse": None,
    },
    {
        "name": "databricks-cli",
        "url": "https://github.com/databricks/cli.git",
        "branch": "main",
        "glob": "**/*.md",
        "collection": "databricks_cli",
        "sparse": None,
    },
]

# Databricks public docs sitemap sections to scrape
DOCS_SITEMAP_URL = "https://docs.databricks.com/aws/en/sitemap.xml"
# Sections to skip (low-value or non-documentation content)
DOCS_SKIP_SECTIONS = [
    "/aws/en/archive/",
    "/aws/en/release-notes/",
    "/aws/en/search/",
    "/aws/en/category/",
]
DOCS_COLLECTION = "databricks_docs"

# Databricks Knowledge Base (troubleshooting articles)
KB_SITEMAP_URL = "https://kb.databricks.com/sitemap.xml"
KB_COLLECTION = "databricks_kb"

CLONE_DIR = Path("/tmp/docs-repos")

# Detect local vs Docker environment
_SCRIPT_DIR = Path(__file__).resolve().parent
_LOCAL_CHROMA = _SCRIPT_DIR / "chroma_data"
_DOCKER_CHROMA = Path("/app/chroma_data")
CHROMA_DIR = _LOCAL_CHROMA if _LOCAL_CHROMA.parent.exists() and not _DOCKER_CHROMA.exists() else _DOCKER_CHROMA

# Model resolution: local model_cache > Docker /app/model > HuggingFace download
_MODEL_DIRS = [
    _SCRIPT_DIR / "model_cache" / "all-mpnet-base-v2",
    Path("/app/model/all-mpnet-base-v2"),
    _SCRIPT_DIR / "model_cache" / "all-MiniLM-L6-v2",
    Path("/app/model/all-MiniLM-L6-v2"),
]
EMBEDDING_MODEL = next((str(d) for d in _MODEL_DIRS if d.exists()), "all-mpnet-base-v2")

# Use MPS (Apple Silicon GPU) when available
def _get_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

# Chunking config
CHUNK_SIZE = 1500  # chars
CHUNK_OVERLAP = 200  # chars
MERGE_TARGET = 1200  # target size when merging small sections

# File patterns to skip in git repos
SKIP_PATTERNS = [
    "CHANGELOG*", "NOTICE*", "LICENSE*",
    ".github/*", "test/*", "tests/*", "__pycache__/*",
]


def _matches_skip_pattern(rel_path: str) -> bool:
    """Check if a relative file path matches any skip pattern."""
    from fnmatch import fnmatch
    rel_path_normalized = rel_path.replace("\\", "/")
    for pattern in SKIP_PATTERNS:
        # Check against the full relative path and each path component
        if fnmatch(rel_path_normalized, pattern):
            return True
        # Also check if any path component matches (e.g. "tests/*" should skip "tests/foo/bar.md")
        parts = rel_path_normalized.split("/")
        for i in range(len(parts)):
            partial = "/".join(parts[i:])
            if fnmatch(partial, pattern):
                return True
    return False


def clean_chunk(text: str) -> str:
    """Clean a chunk of markdown text by removing formatting artifacts and boilerplate."""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Remove markdown table separator lines: | --- | --- | ... |
        if re.match(r'^\|[\s\-:]+(\|[\s\-:]+)+\|?$', stripped):
            continue
        # Remove empty table cell lines: |  |  |  |
        if re.match(r'^\|(\s*\|)+\s*$', stripped):
            continue
        # Remove navigation/TOC cruft: * [Link text](#anchor)
        if re.match(r'^\*\s+\[.*?\]\(#.*?\)\s*$', stripped):
            continue
        # Remove "On this page" boilerplate
        if re.match(r'^(On this page|In this article|Table of contents)\s*$', stripped, re.IGNORECASE):
            continue
        # Remove "Last updated on ..." boilerplate
        if re.match(r'^Last updated on\s', stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    # Collapse excessive whitespace: more than 2 consecutive newlines -> 2
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def extract_page_metadata(text: str, fallback_title: str = "") -> dict:
    """Extract page-level metadata: title and description from markdown text."""
    page_title = fallback_title
    page_description = ""

    # Find the first H1 heading
    h1_match = re.search(r'^#\s+(.+)', text, re.MULTILINE)
    if h1_match:
        page_title = h1_match.group(1).strip()

    # Find the first non-empty paragraph (skip headings, blank lines, and metadata)
    for block in re.split(r'\n\n+', text):
        block = block.strip()
        if not block:
            continue
        # Skip headings
        if block.startswith('#'):
            continue
        # Skip lines that look like metadata (e.g., frontmatter-style)
        if block.startswith('---'):
            continue
        # Skip very short lines that are likely artifacts
        if len(block) < 20:
            continue
        page_description = block[:200]
        break

    return {"page_title": page_title, "page_description": page_description}


def clone_repo(repo: dict) -> Path:
    """Clone a git repo (sparse checkout if specified)."""
    dest = CLONE_DIR / repo["name"]
    if dest.exists():
        shutil.rmtree(dest)

    logger.info(f"Cloning {repo['url']} (branch: {repo['branch']})...")

    if repo.get("sparse"):
        # Sparse checkout — only pull specific dirs
        os.system(f"git clone --depth 1 --filter=blob:none --sparse -b {repo['branch']} {repo['url']} {dest}")
        for path in repo["sparse"]:
            os.system(f"cd {dest} && git sparse-checkout add {path}")
    else:
        os.system(f"git clone --depth 1 -b {repo['branch']} {repo['url']} {dest}")

    return dest


def find_markdown_files(base_dir: Path, glob_pattern: str) -> list[Path]:
    """Find all markdown files matching the glob pattern."""
    files = list(base_dir.glob(glob_pattern))
    logger.info(f"Found {len(files)} markdown files in {base_dir}")
    return files


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, page_meta: dict | None = None) -> Generator[dict, None, None]:
    """Split text into overlapping chunks, preserving section context.

    Small consecutive sections are merged up to MERGE_TARGET chars to produce
    bigger, more contextual chunks (1000-1500 chars typical).
    """
    # Clean up the text
    text = text.strip()
    if not text or len(text) < 50:
        return

    # Extract page-level metadata if not provided
    if page_meta is None:
        page_meta = extract_page_metadata(text, fallback_title=source.rsplit("/", 1)[-1])

    # Try to split on markdown headers first
    raw_sections = re.split(r'\n(?=#{1,3}\s)', text)

    # Track heading hierarchy (h1 > h2 > h3) for each section
    heading_stack = {}  # level -> title
    sections: list[tuple[str, str, str]] = []  # (section_text, section_title, hierarchy)
    for raw_sec in raw_sections:
        raw_sec = raw_sec.strip()
        if not raw_sec:
            continue
        title_match = re.match(r'^(#{1,3})\s+(.+)', raw_sec)
        sec_title = title_match.group(2).strip() if title_match else ""
        sec_level = len(title_match.group(1)) if title_match else 0

        # Update heading stack — clear lower levels when a higher heading appears
        if sec_level > 0:
            heading_stack[sec_level] = sec_title
            for lvl in list(heading_stack):
                if lvl > sec_level:
                    del heading_stack[lvl]

        # Build hierarchy string: "H1 > H2 > H3"
        hierarchy = " > ".join(heading_stack[lvl] for lvl in sorted(heading_stack) if heading_stack[lvl])

        if (
            sections
            and len(sections[-1][0]) + len(raw_sec) + 2 <= MERGE_TARGET
            and not sec_title  # don't merge into a new titled section
        ):
            # Merge with previous section
            prev_text, prev_title, prev_hier = sections[-1]
            sections[-1] = (prev_text + "\n\n" + raw_sec, prev_title, prev_hier)
        elif (
            sections
            and len(sections[-1][0]) < 200
            and len(sections[-1][0]) + len(raw_sec) + 2 <= chunk_size
        ):
            # Previous section is very small — merge even with a new title
            prev_text, prev_title, prev_hier = sections[-1]
            merged_title = prev_title or sec_title
            sections[-1] = (prev_text + "\n\n" + raw_sec, merged_title, hierarchy or prev_hier)
        else:
            sections.append((raw_sec, sec_title, hierarchy))

    global_chunk_idx = 0
    for section, section_title, hierarchy in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= chunk_size:
            # Clean the chunk
            cleaned = clean_chunk(section)
            if not cleaned or len(cleaned) < 20:
                continue
            chunk_id = hashlib.md5(f"{source}:g{global_chunk_idx}:{cleaned[:200]}".encode()).hexdigest()
            yield {
                "id": chunk_id,
                "text": cleaned,
                "source": source,
                "section": section_title,
                "hierarchy": hierarchy,
                "page_title": page_meta.get("page_title", ""),
                "page_description": page_meta.get("page_description", ""),
            }
            global_chunk_idx += 1
        else:
            # Split large sections into overlapping chunks
            start = 0
            while start < len(section):
                end = start + chunk_size

                # Try to break at a paragraph or sentence boundary
                if end < len(section):
                    for boundary in ["\n\n", "\n", ". ", ", "]:
                        bp = section.rfind(boundary, start + chunk_size // 2, end + 100)
                        if bp > start:
                            end = bp + len(boundary)
                            break

                chunk_text_str = section[start:end].strip()
                # Clean the chunk
                chunk_text_str = clean_chunk(chunk_text_str)
                if chunk_text_str and len(chunk_text_str) >= 20:
                    chunk_id = hashlib.md5(f"{source}:g{global_chunk_idx}:{chunk_text_str[:200]}".encode()).hexdigest()
                    yield {
                        "id": chunk_id,
                        "text": chunk_text_str,
                        "source": source,
                        "section": section_title,
                        "hierarchy": hierarchy,
                        "page_title": page_meta.get("page_title", ""),
                        "page_description": page_meta.get("page_description", ""),
                    }
                    global_chunk_idx += 1

                start = end - overlap


def dedup_chunks(chunks: list[dict]) -> list[dict]:
    """Remove chunks with duplicate IDs, keeping the first occurrence."""
    seen = set()
    result = []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            result.append(c)
    dupes = len(chunks) - len(result)
    if dupes:
        logger.info(f"  Removed {dupes} duplicate chunks")
    return result


def embed_and_store(chunks: list[dict], collection, model: SentenceTransformer):
    """Embed and upsert chunks into a ChromaDB collection."""
    chunks = dedup_chunks(chunks)
    if not chunks:
        return

    logger.info(f"Embedding {len(chunks)} chunks...")

    batch_size = 256
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [
            {
                "source": c["source"],
                "section": c["section"],
                "hierarchy": c.get("hierarchy", ""),
                "page_title": c.get("page_title", ""),
                "page_description": c.get("page_description", ""),
            }
            for c in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")


def scrape_databricks_docs(model: SentenceTransformer, client: chromadb.PersistentClient):
    """Scrape docs.databricks.com for key sections and index them."""
    logger.info(f"Fetching sitemap from {DOCS_SITEMAP_URL}...")

    try:
        resp = requests.get(DOCS_SITEMAP_URL, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch sitemap: {e}")
        return

    # Parse sitemap XML
    root = ET.fromstring(resp.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    all_urls = [loc.text for loc in root.findall(".//sm:loc", ns) if loc.text]

    # Index all pages except skipped sections
    urls = [u for u in all_urls if not any(skip in u for skip in DOCS_SKIP_SECTIONS)]
    logger.info(f"Found {len(urls)} pages to index (skipped {len(all_urls) - len(urls)} from excluded sections)")

    # No cap — index all matching pages

    collection = client.get_or_create_collection(
        name=DOCS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    def _scrape_page(url: str) -> list[dict]:
        """Scrape a single page and return its chunks."""
        try:
            page_resp = requests.get(url, timeout=15, headers={"User-Agent": "DatabricksDocsMCP/1.0"})
            if page_resp.status_code != 200:
                return []

            soup = BeautifulSoup(page_resp.text, "html.parser")

            # Remove nav, header, footer, sidebar
            for tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style"]):
                tag.decompose()

            # Try to find main content area
            main = soup.find("main") or soup.find("article") or soup.find("div", {"role": "main"})
            if not main:
                main = soup.body or soup

            # Convert to markdown
            text = md(str(main), strip=["img"]).strip()
            if not text or len(text) < 100:
                return []

            source = url.replace("https://docs.databricks.com/", "docs.databricks.com/")
            # Extract page-level title from URL as fallback
            fallback_title = url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title()
            page_meta = extract_page_metadata(text, fallback_title=fallback_title)
            return list(chunk_text(text, source=source, page_meta=page_meta))

        except Exception as e:
            logger.warning(f"Error scraping {url}: {e}")
            return []

    all_chunks = []
    scraped = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_scrape_page, url): url for url in urls}
        for future in as_completed(futures):
            chunks = future.result()
            all_chunks.extend(chunks)
            scraped += 1
            if scraped % 100 == 0:
                logger.info(f"  Scraped {scraped}/{len(urls)} pages ({len(all_chunks)} chunks)")

    if not all_chunks:
        logger.warning("No chunks from docs.databricks.com scraping")
        return

    embed_and_store(all_chunks, collection, model)
    logger.info(f"Done: docs.databricks.com → {collection.count()} chunks in '{DOCS_COLLECTION}'")


def scrape_databricks_kb(model: SentenceTransformer, client: chromadb.PersistentClient):
    """Scrape kb.databricks.com knowledge base articles and index them."""
    logger.info(f"Fetching KB sitemap from {KB_SITEMAP_URL}...")

    try:
        resp = requests.get(KB_SITEMAP_URL, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to fetch KB sitemap: {e}")
        return

    root = ET.fromstring(resp.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//sm:loc", ns) if loc.text]
    logger.info(f"Found {len(urls)} KB articles")

    collection = client.get_or_create_collection(
        name=KB_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    def _scrape_kb_page(url: str) -> list[dict]:
        try:
            # Use Helpjuice JSON API — append .json to get structured article data
            json_url = url.rstrip("/") + ".json"
            page_resp = requests.get(json_url, timeout=15, headers={"User-Agent": "DatabricksDocsMCP/1.0"})
            if page_resp.status_code != 200:
                return []

            data = page_resp.json()
            html_body = data.get("answer") or data.get("processed_answer") or ""
            if not html_body:
                return []

            # Convert HTML article body to markdown
            text = md(html_body, strip=["img"]).strip()
            if not text or len(text) < 100:
                return []

            title = data.get("name", "")
            description = data.get("description", "")
            source = url.replace("https://kb.databricks.com/", "kb.databricks.com/")
            fallback_title = url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title()
            page_meta = {
                "page_title": title or fallback_title,
                "page_description": description[:200] if description else "",
            }
            return list(chunk_text(text, source=source, page_meta=page_meta))

        except Exception as e:
            logger.warning(f"Error scraping KB {url}: {e}")
            return []

    all_chunks = []
    scraped = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_scrape_kb_page, url): url for url in urls}
        for future in as_completed(futures):
            chunks = future.result()
            all_chunks.extend(chunks)
            scraped += 1
            if scraped % 100 == 0:
                logger.info(f"  Scraped {scraped}/{len(urls)} KB articles ({len(all_chunks)} chunks)")

    if not all_chunks:
        logger.warning("No chunks from kb.databricks.com scraping")
        return

    embed_and_store(all_chunks, collection, model)
    logger.info(f"Done: kb.databricks.com → {collection.count()} chunks in '{KB_COLLECTION}'")


def process_repo(repo: dict, model: SentenceTransformer, client: chromadb.PersistentClient):
    """Clone, chunk, embed, and store a single repo's docs."""
    repo_dir = clone_repo(repo)
    files = find_markdown_files(repo_dir, repo["glob"])

    if not files:
        logger.warning(f"No files found for {repo['name']}, skipping")
        return

    collection = client.get_or_create_collection(
        name=repo["collection"],
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    skipped = 0
    for md_file in files:
        # Relative path for source reference
        rel_path = str(md_file.relative_to(repo_dir))

        # Skip files matching excluded patterns
        if _matches_skip_pattern(rel_path):
            skipped += 1
            continue

        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Error reading {md_file}: {e}")
            continue

        # Extract page metadata using filename as fallback title
        fallback_title = md_file.stem.replace("-", " ").replace("_", " ").title()
        page_meta = extract_page_metadata(text, fallback_title=fallback_title)

        for chunk in chunk_text(text, source=f"{repo['name']}/{rel_path}", page_meta=page_meta):
            all_chunks.append(chunk)

    if skipped:
        logger.info(f"  Skipped {skipped} files matching exclude patterns")

    if not all_chunks:
        logger.warning(f"No chunks produced for {repo['name']}")
        return

    embed_and_store(all_chunks, collection, model)
    logger.info(f"Done: {repo['name']} → {collection.count()} chunks in '{repo['collection']}'")


def collection_exists(client: chromadb.PersistentClient, name: str) -> bool:
    """Check if a collection already has data."""
    try:
        col = client.get_collection(name)
        return col.count() > 0
    except Exception:
        return False


def main():
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # --force flag to reindex everything
    force = "--force" in sys.argv

    device = _get_device()
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL} (device: {device})")
    logger.info(f"ChromaDB path: {CHROMA_DIR}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # When forcing, delete all existing collections first
    # (required when embedding model dimension changes)
    if force:
        for col in client.list_collections():
            logger.info(f"Force: deleting collection '{col.name}'")
            client.delete_collection(col.name)

    # Index GitHub repos (skip if already indexed)
    for repo in REPOS:
        if not force and collection_exists(client, repo["collection"]):
            logger.info(f"Skipping {repo['name']} — already indexed")
            continue
        try:
            process_repo(repo, model, client)
        except Exception as e:
            logger.error(f"Error processing {repo['name']}: {e}")

    # Scrape and index docs.databricks.com
    if force or not collection_exists(client, DOCS_COLLECTION):
        try:
            scrape_databricks_docs(model, client)
        except Exception as e:
            logger.error(f"Error scraping docs.databricks.com: {e}")
    else:
        logger.info("Skipping docs.databricks.com — already indexed")

    # Scrape and index kb.databricks.com
    if force or not collection_exists(client, KB_COLLECTION):
        try:
            scrape_databricks_kb(model, client)
        except Exception as e:
            logger.error(f"Error scraping kb.databricks.com: {e}")
    else:
        logger.info("Skipping kb.databricks.com — already indexed")

    # Print summary
    logger.info("=== Index Summary ===")
    for col in client.list_collections():
        logger.info(f"  {col.name}: {col.count()} chunks")

    # Cleanup cloned repos
    shutil.rmtree(CLONE_DIR, ignore_errors=True)
    logger.info("Build complete.")


if __name__ == "__main__":
    main()
