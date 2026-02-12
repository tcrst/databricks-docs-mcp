"""
Databricks Documentation MCP Server.

Exposes semantic search over Databricks docs, API reference, and Terraform provider
docs via MCP tools. Connects to Claude Code via stdio transport.

Usage:
    python server.py                    # stdio (for Claude Code)
    python server.py --transport sse    # SSE (for remote clients)
"""

import json
import logging
from pathlib import Path

import requests as http_requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent

# Resolve paths: local first, then Docker fallback
_LOCAL_CHROMA = _SCRIPT_DIR / "chroma_data"
_DOCKER_CHROMA = Path("/app/chroma_data")
CHROMA_DIR = _LOCAL_CHROMA if _LOCAL_CHROMA.exists() else _DOCKER_CHROMA

# Prefer mpnet (768-dim, higher quality) if available, fall back to MiniLM
_MODEL_DIRS = [
    _SCRIPT_DIR / "model_cache" / "all-mpnet-base-v2",
    Path("/app/model/all-mpnet-base-v2"),
    _SCRIPT_DIR / "model_cache" / "all-MiniLM-L6-v2",
    Path("/app/model/all-MiniLM-L6-v2"),
]
EMBEDDING_MODEL = next((str(d) for d in _MODEL_DIRS if d.exists()), "all-mpnet-base-v2")
DEFAULT_RESULTS = 5

# Sources to deprioritize in search results (non-documentation files)
NOISE_SOURCES = {"CHANGELOG.md", "NOTICE.md", "LICENSE.md", "LICENSE", "CONTRIBUTING.md",
                 "CODE_OF_CONDUCT.md", ".gitignore"}

# Use MPS (Apple Silicon GPU) when available for faster embedding
def _get_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

# Initialize once at module level
_device = _get_device()
logger.info(f"Loading embedding model: {EMBEDDING_MODEL} (device: {_device})")
model = SentenceTransformer(EMBEDDING_MODEL, device=_device)

# Cross-encoder for reranking (much more accurate than bi-encoder similarity)
_RERANKER_DIRS = [
    _SCRIPT_DIR / "model_cache" / "ms-marco-MiniLM-L-6-v2",
    Path("/app/model/ms-marco-MiniLM-L-6-v2"),
]
_RERANKER_MODEL = next((str(d) for d in _RERANKER_DIRS if d.exists()), "cross-encoder/ms-marco-MiniLM-L-6-v2")
logger.info(f"Loading reranker model: {_RERANKER_MODEL}")
reranker = CrossEncoder(_RERANKER_MODEL, device=_device)

logger.info("Connecting to ChromaDB...")
client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)

# Load collections
collections = {}
for col in client.list_collections():
    collections[col.name] = col
    logger.info(f"Loaded collection: {col.name} ({col.count()} chunks)")

mcp = FastMCP(
    "Databricks Docs",
    instructions=(
        "Search Databricks documentation, API reference, and Terraform provider docs. "
        "Use these tools to find relevant documentation when working with Databricks, "
        "Genie API, Unity Catalog, SQL warehouses, or the Databricks Terraform provider."
    ),
)


MIN_RELEVANCE = 0.35
MAX_CONTENT_LEN = 500

# Keywords that map to URL path segments for keyword boosting
KEYWORD_TO_PATH = {
    "genie": "genie", "ai/bi": "genie", "aibi": "genie",
    "unity catalog": "unity-catalog", "unity-catalog": "unity-catalog",
    "sql warehouse": "warehouses", "warehouse": "warehouses",
    "terraform": "terraform", "cluster": "cluster",
    "notebook": "notebook", "job": "job", "pipeline": "pipeline",
    "delta": "delta", "mlflow": "machine-learning",
    "workspace": "workspace", "secret": "secret", "token": "token",
}
KEYWORD_BOOST = 0.08  # boost applied when source matches query keyword


def _source_to_url(source: str) -> str:
    """Convert a source file path to a documentation URL."""
    if source.startswith("https://"):
        return source
    if source.startswith("http://"):
        return source
    # Bare domain paths like "docs.databricks.com/aws/en/..."
    if source.startswith("docs.databricks.com") or source.startswith("kb.databricks.com"):
        return f"https://{source}"
    if source.startswith("terraform-provider-databricks/docs/"):
        path = source.replace("terraform-provider-databricks/docs/", "")
        path = path.replace(".md", "")
        return f"https://registry.terraform.io/providers/databricks/databricks/latest/docs/{path}"
    if source.startswith("databricks-sdk-py/docs/"):
        path = source.replace("databricks-sdk-py/docs/", "")
        path = path.replace(".md", "").replace(".rst", "")
        return f"https://databricks-sdk-py.readthedocs.io/en/latest/{path}"
    if source.startswith("databricks-cli/"):
        path = source.replace("databricks-cli/", "")
        return f"https://github.com/databricks/cli/blob/main/{path}"
    if source.startswith("genai-cookbook/"):
        path = source.replace("genai-cookbook/", "")
        return f"https://github.com/databricks/genai-cookbook/blob/main/{path}"
    return source


def _truncate(text: str, max_len: int = MAX_CONTENT_LEN) -> str:
    """Truncate text at a word boundary."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + " ..."


def _format_results(results: list[dict], collection_label: str | None = None) -> str:
    """Format search results as readable markdown."""
    if not results:
        label = f" in {collection_label}" if collection_label else ""
        return f"No relevant results found{label}."

    lines = []
    for i, r in enumerate(results, 1):
        section = r.get("section", "")
        page_title = r.get("page_title", "")
        source = r.get("source", "unknown")
        url = _source_to_url(source)
        relevance = r.get("relevance", 0)
        content = _truncate(r.get("content", ""))
        collection = r.get("collection")

        title = section if section else page_title if page_title else source.split("/")[-1]
        source_parts = [f"[Source]({url})"]
        if collection:
            source_parts.append(f"Collection: `{collection}`")
        source_parts.append(f"Relevance: {relevance:.0%}")

        lines.append(f"### {i}. {title}")
        lines.append(" | ".join(source_parts))
        lines.append("")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


def _extract_boost_paths(query: str) -> list[str]:
    """Extract URL path segments that should be boosted based on query keywords."""
    q_lower = query.lower()
    paths = []
    for keyword, path in KEYWORD_TO_PATH.items():
        if keyword in q_lower and path not in paths:
            paths.append(path)
    return paths


def _search(collection_name: str, query: str, n_results: int = DEFAULT_RESULTS,
            *, raw: bool = False) -> list[dict] | str:
    """Perform semantic search on a collection.

    Uses semantic search with keyword-aware boosting: results whose source URL
    contains terms from the query get a relevance boost to surface domain-specific
    pages that might otherwise rank below generic matches.

    Args:
        raw: If True, return list[dict] for programmatic use (e.g. search_all merging).
             If False, return formatted markdown string.
    """
    col = collections.get(collection_name)
    if not col or col.count() == 0:
        if raw:
            return []
        return f"Collection '{collection_name}' not found or empty."

    query_embedding = model.encode([query]).tolist()

    # Fetch more candidates to improve recall (3x requested, minimum 20)
    fetch_n = min(max(n_results * 3, 20), col.count())
    results = col.query(
        query_embeddings=query_embedding,
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    # Determine keyword boost paths from the query
    boost_paths = _extract_boost_paths(query)

    output = []
    for i in range(len(results["ids"][0])):
        relevance = round(1 - results["distances"][0][i], 3)
        if relevance < MIN_RELEVANCE:
            continue
        source = results["metadatas"][0][i].get("source", "unknown")
        # Skip noise sources (changelogs, licenses, etc.)
        source_filename = source.rsplit("/", 1)[-1] if "/" in source else source
        if source_filename in NOISE_SOURCES:
            continue

        # Apply keyword boost when source URL matches query terms
        boosted_relevance = relevance
        if boost_paths:
            source_lower = source.lower()
            for bp in boost_paths:
                if bp in source_lower:
                    boosted_relevance = min(relevance + KEYWORD_BOOST, 1.0)
                    break

        output.append({
            "content": results["documents"][0][i],
            "source": source,
            "section": results["metadatas"][0][i].get("section", ""),
            "page_title": results["metadatas"][0][i].get("page_title", ""),
            "relevance": relevance,
            "boosted_relevance": boosted_relevance,
        })

    # Sort by boosted relevance to surface keyword-matched results
    output.sort(key=lambda x: x["boosted_relevance"], reverse=True)

    # Take top candidates for cross-encoder reranking
    rerank_pool = output[:max(n_results * 3, 15)]

    if len(rerank_pool) > 1:
        pairs = [[query, item["content"]] for item in rerank_pool]
        scores = reranker.predict(pairs)
        for item, score in zip(rerank_pool, scores):
            # Normalize cross-encoder score to 0-1 range (sigmoid-like)
            norm_score = 1 / (1 + __import__("math").exp(-score))
            # Blend: 60% cross-encoder, 40% bi-encoder (boosted)
            item["relevance"] = round(0.6 * norm_score + 0.4 * item["boosted_relevance"], 3)
        rerank_pool.sort(key=lambda x: x["relevance"], reverse=True)
        output = rerank_pool
    else:
        for item in output:
            item["relevance"] = item.pop("boosted_relevance")

    # Clean up internal field
    for item in output:
        item.pop("boosted_relevance", None)

    output = output[:n_results]

    if raw:
        return output
    return _format_results(output, collection_name)


@mcp.tool
def search_databricks_docs(query: str, n_results: int = 5) -> str:
    """Search Databricks general documentation.

    Use this for questions about Databricks features, setup, configuration,
    Genie, Unity Catalog, SQL warehouses, notebooks, clusters, jobs, etc.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)
    """
    return _search("databricks_docs", query, n_results)


@mcp.tool
def search_databricks_api(query: str, n_results: int = 5) -> str:
    """Search Databricks Python SDK and API documentation.

    Use this for questions about Databricks REST API endpoints, Python SDK
    methods, authentication, client configuration, and programmatic access.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)
    """
    return _search("databricks_api", query, n_results)


@mcp.tool
def search_terraform_databricks(query: str, n_results: int = 5) -> str:
    """Search Databricks Terraform provider documentation.

    Use this for questions about Terraform resources for Databricks: workspaces,
    clusters, jobs, permissions, Unity Catalog, secrets, tokens, and IaC patterns.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)
    """
    return _search("terraform_databricks", query, n_results)


@mcp.tool
def list_collections() -> list[dict]:
    """List all available documentation collections and their sizes.

    Returns the name and chunk count for each indexed documentation source.
    """
    return [
        {"name": name, "chunks": col.count()}
        for name, col in collections.items()
    ]


@mcp.tool
def search_databricks_genai(query: str, n_results: int = 5) -> str:
    """Search Databricks GenAI Cookbook documentation.

    Use this for questions about building RAG applications, GenAI best practices,
    evaluation strategies, chunking, retrieval, and LLM application patterns
    on Databricks.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)
    """
    return _search("databricks_genai", query, n_results)


@mcp.tool
def search_databricks_cli(query: str, n_results: int = 5) -> str:
    """Search Databricks CLI documentation.

    Use this for questions about the Databricks command-line interface,
    CLI commands, bundle deployment, authentication setup, and workspace management.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)
    """
    return _search("databricks_cli", query, n_results)


@mcp.tool
def search_databricks_kb(query: str, n_results: int = 5) -> str:
    """Search Databricks Knowledge Base (troubleshooting articles).

    Use this for questions about common errors, troubleshooting steps,
    known issues, workarounds, and best practices from the Databricks KB.

    Args:
        query: Natural language search query
        n_results: Number of results to return (default: 5)
    """
    return _search("databricks_kb", query, n_results)


@mcp.tool
def search_all(query: str, n_results: int = 5) -> str:
    """Search across ALL documentation sources and return a unified, ranked list.

    Queries every collection, deduplicates similar results, and returns the top
    results sorted by relevance. Each result includes which source it came from.
    Use this when you're unsure which source is most relevant, or when you want
    a comprehensive answer that draws from multiple documentation sources.

    Args:
        query: Natural language search query
        n_results: Total number of top results to return (default: 5)
    """
    candidates_per_source = max(3, n_results)

    all_results = []
    for name in collections:
        results = _search(name, query, candidates_per_source, raw=True)
        for r in results:
            r["collection"] = name
            all_results.append(r)

    all_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

    # Deduplicate by word overlap
    seen_content = []
    deduped = []
    for result in all_results:
        content_preview = result["content"][:200].strip().lower()
        is_duplicate = False
        for seen in seen_content:
            words_a = set(content_preview.split())
            words_b = set(seen.split())
            if len(words_a & words_b) > 0.6 * max(len(words_a), len(words_b), 1):
                is_duplicate = True
                break
        if not is_duplicate:
            seen_content.append(content_preview)
            deduped.append(result)

    return _format_results(deduped[:n_results])


@mcp.tool
def fetch_page(source: str, collection: str | None = None) -> str:
    """Fetch the FULL content of a documentation page.

    Use this after searching to read a complete page — all chunks reassembled
    in order. This is essential for learning, as search only returns snippets.

    Pass the `source` value from a search result. Optionally specify the
    collection name to speed up the lookup.

    Args:
        source: The source path or URL from a search result
        collection: Optional collection name (searches all if omitted)
    """
    target_collections = (
        [collections[collection]] if collection and collection in collections
        else list(collections.values())
    )

    all_chunks = []
    for col in target_collections:
        # ChromaDB where filter on metadata
        try:
            results = col.get(
                where={"source": source},
                include=["documents", "metadatas"],
            )
        except Exception:
            continue

        if results and results["ids"]:
            for i in range(len(results["ids"])):
                section = results["metadatas"][i].get("section", "")
                all_chunks.append({
                    "id": results["ids"][i],
                    "section": section,
                    "content": results["documents"][i],
                })

    if not all_chunks:
        return f"No page found for source: `{source}`\n\nTip: copy the exact `source` value from a search result."

    # Sort by chunk ID to preserve document order
    all_chunks.sort(key=lambda c: c["id"])

    url = _source_to_url(source)
    lines = [f"# {source.split('/')[-1]}", f"**URL:** {url}", f"**Chunks:** {len(all_chunks)}", ""]

    seen_sections = set()
    for chunk in all_chunks:
        section = chunk["section"]
        if section and section not in seen_sections:
            lines.append(f"## {section}")
            seen_sections.add(section)
        lines.append(chunk["content"].strip())
        lines.append("")

    return "\n".join(lines)


@mcp.tool
def list_pages(collection: str, query: str | None = None, limit: int = 30) -> str:
    """List available pages/documents in a collection for browsing.

    Returns unique source pages with their sections. Use this to explore
    what documentation is available before searching for specifics.
    Optionally filter by a keyword in the source path or section names.

    Args:
        collection: Collection name (e.g. 'databricks_docs', 'terraform_databricks')
        query: Optional keyword to filter page paths (e.g. 'genie', 'unity-catalog')
        limit: Max pages to return (default: 30)
    """
    col = collections.get(collection)
    if not col:
        names = ", ".join(f"`{n}`" for n in collections)
        return f"Collection `{collection}` not found. Available: {names}"

    # Get all metadata to extract unique sources
    all_meta = col.get(include=["metadatas"])
    if not all_meta or not all_meta["ids"]:
        return f"Collection `{collection}` is empty."

    # Group sections by source
    pages: dict[str, list[str]] = {}
    for meta in all_meta["metadatas"]:
        source = meta.get("source", "unknown")
        section = meta.get("section", "")
        if source not in pages:
            pages[source] = []
        if section and section not in pages[source]:
            pages[source].append(section)

    # Filter by query if provided
    if query:
        q = query.lower()
        pages = {
            src: sections for src, sections in pages.items()
            if q in src.lower() or any(q in s.lower() for s in sections)
        }

    if not pages:
        return f"No pages matching `{query}` in `{collection}`."

    # Sort by source path and limit
    sorted_pages = sorted(pages.items())[:limit]

    lines = [f"## {collection} — {len(pages)} pages" + (f" matching `{query}`" if query else "")]
    lines.append("")

    for source, sections in sorted_pages:
        url = _source_to_url(source)
        lines.append(f"- **[{source.split('/')[-1]}]({url})**")
        if sections:
            for s in sections[:5]:
                lines.append(f"  - {s}")
            if len(sections) > 5:
                lines.append(f"  - ... +{len(sections) - 5} more sections")

    if len(pages) > limit:
        lines.append(f"\n*Showing {limit} of {len(pages)} pages. Use `query` to filter.*")

    return "\n".join(lines)


def _get_section_chunks(source: str, section: str, col) -> list[dict]:
    """Get all chunks belonging to a specific source+section, ordered."""
    try:
        results = col.get(
            where={"$and": [{"source": source}, {"section": section}]},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    chunks = []
    if results and results["ids"]:
        for i in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][i],
                "content": results["documents"][i],
            })
    chunks.sort(key=lambda c: c["id"])
    return chunks


def _collection_label(name: str) -> str:
    """Human-friendly label for a collection name."""
    labels = {
        "databricks_docs": "Databricks Docs",
        "databricks_api": "Python SDK / API",
        "terraform_databricks": "Terraform Provider",
        "databricks_genai": "GenAI Cookbook",
        "databricks_cli": "CLI Reference",
        "databricks_kb": "Knowledge Base",
    }
    return labels.get(name, name)


RESEARCH_MIN_RELEVANCE = 0.40


def _split_compound_query(topic: str) -> list[str]:
    """Split compound queries into independent sub-queries.

    Handles patterns like:
      - "what is X and how to Y" → ["what is X", "how to Y"]
      - "X and how to configure it with Z" → ["X", "how to configure X with Z"]
      - "explain X, Y, and Z" → single query (not compound)
    """
    # Connectors that indicate a compound question
    compound_patterns = [
        r'\band\s+how\s+(?:to|do|does|can)\b',
        r'\band\s+what\s+(?:is|are)\b',
        r'\band\s+how\s+(?:is|are)\b',
        r'\band\s+explain\b',
        r'\band\s+describe\b',
    ]

    import re
    for pattern in compound_patterns:
        match = re.search(pattern, topic, re.IGNORECASE)
        if match:
            part1 = topic[:match.start()].strip().rstrip(",")
            part2 = topic[match.start():].strip()
            # Remove leading "and " from part2
            part2 = re.sub(r'^and\s+', '', part2, flags=re.IGNORECASE)

            # Extract the main subject from part1 for context in part2
            # e.g., "what is a Genie space" → subject "Genie space"
            subject = part1.lower()
            for prefix in ["what is a ", "what is an ", "what is ", "what are ",
                           "explain ", "describe ", "tell me about "]:
                if subject.startswith(prefix):
                    subject = subject[len(prefix):]
                    break
            subject = subject.strip("? ").strip()

            # If part2 uses pronouns like "it", replace with subject
            part2 = re.sub(r'\bit\b', subject, part2, flags=re.IGNORECASE)

            return [part1.strip(), part2.strip()]

    return [topic]


OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:1b"

def _ollama_available() -> bool:
    """Check if Ollama is running."""
    try:
        http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=1)
        return True
    except Exception:
        return False

_ollama_ok = _ollama_available()
if _ollama_ok:
    logger.info(f"Ollama available — using {OLLAMA_MODEL} for query expansion")
else:
    logger.info("Ollama not available — using static query expansion")


def _expand_query_ollama(topic: str) -> list[str]:
    """Use Ollama to generate 2-3 alternate phrasings for better search recall."""
    try:
        resp = http_requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": (
                    "Rephrase this Databricks documentation search query 3 different ways. "
                    "Return ONLY a JSON array of strings, nothing else.\n\n"
                    f"Query: {topic}\n\nJSON array:"
                ),
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 150},
            },
            timeout=15,
        )
        text = resp.json().get("response", "").strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # Extract JSON array from response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            variants = json.loads(text[start:end])
            return [v.strip() for v in variants if isinstance(v, str) and v.strip()][:3]
    except Exception as e:
        logger.debug(f"Ollama expansion failed: {e}")
    return []


def _generate_query_variants(topic: str) -> list[str]:
    """Generate search query variants to improve recall.

    Uses Ollama (if available) for LLM-powered expansion, with static
    fallback for keyword variants.
    """
    queries = [topic]

    # LLM-powered expansion via Ollama
    if _ollama_ok:
        llm_variants = _expand_query_ollama(topic)
        queries.extend(llm_variants)

    # Static fallback: strip question words for a keyword-focused variant
    keyword_variant = topic.lower()
    for prefix in ["what is a ", "what is an ", "what is ", "how to ", "how do i ",
                   "how does ", "explain ", "what are ", "tell me about ", "describe "]:
        if keyword_variant.startswith(prefix):
            keyword_variant = keyword_variant[len(prefix):]
            break
    keyword_variant = keyword_variant.strip("? ").strip()
    if keyword_variant != topic.lower():
        queries.append(keyword_variant)

    # Add "Databricks" prefix if not already present
    if "databricks" not in topic.lower():
        queries.append(f"Databricks {keyword_variant}")

    # Add an "overview introduction" variant for "what is" questions
    if any(topic.lower().startswith(p) for p in ["what is", "what are", "explain", "describe"]):
        queries.append(f"{keyword_variant} overview introduction features")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower not in seen:
            seen.add(q_lower)
            unique.append(q)

    return unique[:6]  # cap at 6 variants


def _keyword_search(collection_name: str, keyword: str, query: str,
                     n_results: int = DEFAULT_RESULTS) -> list[dict]:
    """Search within chunks that contain a specific keyword.

    Uses ChromaDB's where_document filter to find chunks containing the keyword,
    then ranks them by semantic similarity to the query. This is useful as a
    fallback when pure semantic search misses product-specific pages.
    """
    col = collections.get(collection_name)
    if not col or col.count() == 0:
        return []

    query_embedding = model.encode([query]).tolist()

    try:
        results = col.query(
            query_embeddings=query_embedding,
            n_results=min(n_results * 2, col.count()),
            where_document={"$contains": keyword},
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    if not results["ids"] or not results["ids"][0]:
        return []

    output = []
    for i in range(len(results["ids"][0])):
        relevance = round(1 - results["distances"][0][i], 3)
        if relevance < MIN_RELEVANCE:
            continue
        source = results["metadatas"][0][i].get("source", "unknown")
        source_filename = source.rsplit("/", 1)[-1] if "/" in source else source
        if source_filename in NOISE_SOURCES:
            continue
        output.append({
            "content": results["documents"][0][i],
            "source": source,
            "section": results["metadatas"][0][i].get("section", ""),
            "page_title": results["metadatas"][0][i].get("page_title", ""),
            "relevance": relevance,
        })

    return output[:n_results]


def _extract_key_terms(topic: str) -> list[str]:
    """Extract significant proper nouns / product terms from a topic string."""
    # Known Databricks product terms to look for
    known_terms = [
        "Genie", "Unity Catalog", "MLflow", "Delta", "Mosaic",
        "Databricks SQL", "Lakehouse", "Photon", "AutoML",
    ]
    terms = []
    topic_lower = topic.lower()
    for term in known_terms:
        if term.lower() in topic_lower:
            terms.append(term)
    return terms


def _dedup_hits(hits: list[dict]) -> list[dict]:
    """Deduplicate search hits by content overlap."""
    seen = []
    unique = []
    for hit in hits:
        preview = hit["content"][:200].strip().lower()
        is_dup = False
        for s in seen:
            words_a, words_b = set(preview.split()), set(s.split())
            if len(words_a & words_b) > 0.6 * max(len(words_a), len(words_b), 1):
                is_dup = True
                break
        if not is_dup:
            seen.append(preview)
            unique.append(hit)
    return unique


@mcp.tool
def research(topic: str, depth: str = "standard") -> str:
    """Deep research on a Databricks topic.

    Searches across ALL documentation sources using multiple query variants,
    gathers full sections for the best matches, groups findings by source page,
    deduplicates, and returns a structured research brief organized by subtopic.

    This is the recommended tool for learning about a Databricks topic.
    Use it instead of individual search tools when you want comprehensive,
    well-structured findings.

    Args:
        topic: The topic or question to research (e.g. "Genie API authentication",
               "Unity Catalog permissions model", "how to set up RAG on Databricks")
        depth: "quick" (top 5 snippets), "standard" (top 10 with full sections),
               "deep" (top 20 with full sections from all sources)
    """
    if depth == "quick":
        n_search = 5
        expand_sections = False
    elif depth == "deep":
        n_search = 20
        expand_sections = True
    else:
        n_search = 10
        expand_sections = True

    # Step 1: Split compound queries and generate variants for each sub-query
    sub_queries = _split_compound_query(topic)
    all_query_variants = []
    for sq in sub_queries:
        all_query_variants.extend(_generate_query_variants(sq))
    # Deduplicate while preserving order
    seen_q = set()
    query_variants = []
    for q in all_query_variants:
        q_lower = q.lower()
        if q_lower not in seen_q:
            seen_q.add(q_lower)
            query_variants.append(q)

    candidates_per_query = max(5, n_search)

    all_hits: dict[str, dict] = {}  # keyed by (source, section) to dedup across queries

    # Phase 1: Semantic search with keyword boosting
    for query in query_variants:
        for col_name in collections:
            hits = _search(col_name, query, candidates_per_query, raw=True)
            for h in hits:
                if h["relevance"] < RESEARCH_MIN_RELEVANCE:
                    continue
                h["collection"] = col_name
                # Keep best relevance per unique chunk
                key = f"{h['source']}::{h.get('section', '')}::{h['content'][:100]}"
                if key not in all_hits or h["relevance"] > all_hits[key]["relevance"]:
                    all_hits[key] = h

    # Phase 2: Keyword-filtered search for product-specific terms
    # This catches pages that semantic search might miss (e.g., "Genie" pages)
    key_terms = _extract_key_terms(topic)
    for term in key_terms:
        for col_name in collections:
            for query in query_variants[:2]:  # use top 2 query variants
                kw_hits = _keyword_search(col_name, term, query, n_results=candidates_per_query)
                for h in kw_hits:
                    if h["relevance"] < RESEARCH_MIN_RELEVANCE:
                        continue
                    h["collection"] = col_name
                    key = f"{h['source']}::{h.get('section', '')}::{h['content'][:100]}"
                    if key not in all_hits or h["relevance"] > all_hits[key]["relevance"]:
                        all_hits[key] = h

    sorted_hits = sorted(all_hits.values(), key=lambda x: x["relevance"], reverse=True)
    unique_hits = _dedup_hits(sorted_hits)
    top_hits = unique_hits[:n_search]

    if not top_hits:
        return f"No relevant documentation found for: **{topic}**"

    # Step 2: Group by source page
    pages: dict[tuple[str, str], dict] = {}
    for hit in top_hits:
        key = (hit["source"], hit["collection"])
        if key not in pages:
            page_title = hit.get("page_title", "")
            pages[key] = {
                "source": hit["source"],
                "collection": hit["collection"],
                "url": _source_to_url(hit["source"]),
                "page_title": page_title,
                "max_relevance": hit["relevance"],
                "sections": {},
            }
        else:
            pages[key]["max_relevance"] = max(pages[key]["max_relevance"], hit["relevance"])

        section = hit.get("section", "") or "Overview"

        if section not in pages[key]["sections"]:
            if expand_sections:
                col = collections.get(hit["collection"])
                if col and hit.get("section"):
                    section_chunks = _get_section_chunks(hit["source"], hit["section"], col)
                    if section_chunks:
                        pages[key]["sections"][section] = "\n".join(
                            c["content"].strip() for c in section_chunks
                        )
                    else:
                        pages[key]["sections"][section] = hit["content"]
                else:
                    pages[key]["sections"][section] = hit["content"]
            else:
                pages[key]["sections"][section] = _truncate(hit["content"], 600)

    # Step 3: Sort pages by relevance
    sorted_pages = sorted(pages.values(), key=lambda p: p["max_relevance"], reverse=True)

    # Step 4: Build structured output
    lines = []
    lines.append(f"# Research: {topic}")
    lines.append(f"*Searched {len(query_variants)} query variants across "
                 f"{len(collections)} collections — found {len(sorted_pages)} relevant pages*")
    lines.append("")

    # Table of contents
    lines.append("## Sources")
    for i, page in enumerate(sorted_pages, 1):
        label = _collection_label(page["collection"])
        page_title = page.get("page_title", "")
        page_name = page_title if page_title else (page["source"].split("/")[-1] or page["source"])
        lines.append(f"{i}. **[{page_name}]({page['url']})** ({label}) — "
                     f"relevance: {page['max_relevance']:.0%}")
    lines.append("")

    # Detailed findings per page
    lines.append("---")
    lines.append("## Findings")
    lines.append("")

    for i, page in enumerate(sorted_pages, 1):
        label = _collection_label(page["collection"])
        page_title = page.get("page_title", "")
        page_name = page_title if page_title else (page["source"].split("/")[-1] or page["source"])
        lines.append(f"### {i}. {page_name}")
        lines.append(f"*Source: {label}* | [Open]({page['url']})")
        lines.append("")

        for section_name, content in page["sections"].items():
            if section_name != "Overview":
                lines.append(f"#### {section_name}")
            clean = content.strip()
            if len(clean) > 2000:
                clean = clean[:2000].rsplit("\n", 1)[0] + "\n\n*(truncated — use `fetch_page` for full content)*"
            lines.append(clean)
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]

    mcp.run(transport=transport)
