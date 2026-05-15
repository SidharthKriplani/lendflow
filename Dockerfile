FROM python:3.11-slim

WORKDIR /app

# System deps for spaCy + Presidio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model for Presidio (graceful — pipeline works without it)
RUN python -m spacy download en_core_web_lg || echo "spaCy model download skipped"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/chroma_db /app/audit_logs /app/tests/fixtures

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run indexer first (builds RAG), then start API
CMD ["sh", "-c", "python rag/indexer.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
