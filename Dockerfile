FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies (no git needed — indexing happens locally)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image (no HuggingFace download at runtime)
COPY model_cache/all-mpnet-base-v2 /app/model/all-mpnet-base-v2
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV SENTENCE_TRANSFORMERS_HOME=/app/model

# Bake pre-built ChromaDB index (built locally with GPU acceleration)
COPY chroma_data /app/chroma_data

# Copy server only (indexer runs locally, not in container)
COPY server.py ./

# stdio transport by default (for Claude Code)
CMD ["python", "server.py"]
