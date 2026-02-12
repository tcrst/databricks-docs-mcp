# Databricks Docs MCP Server

Semantic search over Databricks documentation, API reference, Terraform provider docs, and knowledge base — as an MCP server for Claude Code.

Uses `all-mpnet-base-v2` embeddings + ChromaDB with keyword boosting for high-relevance results.

## Setup

```bash
cd tools/databricks-docs-mcp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build the Index

```bash
python indexer.py
```

Clones doc repos, scrapes docs.databricks.com + kb.databricks.com, chunks, embeds, and stores in `chroma_data/`. Takes ~15 min on first run (uses MPS GPU on Apple Silicon). Skips already-indexed collections on re-run.

## Connect to Claude Code

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "databricks-docs": {
      "command": "/path/to/tools/databricks-docs-mcp/.venv/bin/python",
      "args": ["/path/to/tools/databricks-docs-mcp/server.py"]
    }
  }
}
```

Run `/mcp` in Claude Code to connect.

## Tools

| Tool | What it searches |
|------|-----------------|
| `search_databricks_docs` | General docs — Genie, Unity Catalog, warehouses, etc. |
| `search_databricks_api` | REST API & Python SDK reference |
| `search_terraform_databricks` | Terraform provider resources |
| `search_databricks_genai` | GenAI Cookbook — RAG, evaluation, chunking |
| `search_databricks_cli` | CLI commands, bundles, workspace ops |
| `search_databricks_kb` | Troubleshooting & known issues |
| `search_all` | All collections at once |
| `research` | Deep multi-query research on a topic |
| `list_collections` | Show collections and chunk counts |
| `list_pages` | List indexed pages in a collection |
| `fetch_page` | Get all chunks for a specific page |

## Doc Sources

| Source | Collection | ~Chunks |
|--------|-----------|---------|
| docs.databricks.com | `databricks_docs` | 14,200 |
| kb.databricks.com | `databricks_kb` | 1,100 |
| terraform-provider-databricks | `terraform_databricks` | 2,200 |
| databricks-sdk-py | `databricks_api` | 107 |
| genai-cookbook | `databricks_genai` | 176 |
| databricks CLI | `databricks_cli` | 689 |
