#!/bin/bash
set -e

# Run indexer — it skips collections that are already populated,
# so this is fast on subsequent runs (only indexes new sources)
# Pass --force to re-index everything (deletes existing collections first)
python indexer.py "$@"

# Start MCP server (stdio transport by default)
exec python server.py
