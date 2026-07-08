# Headless REST API server image (FastAPI, `rag-app serve`).
# The PySide6 desktop GUI is deliberately excluded — it's a desktop app and
# lives behind the optional [gui] extra.
#
# Build:            docker build -t rag-api .
# Build with OCR:   docker build -t rag-api --build-arg WITH_OCR=true .
# Build with Qdrant client: docker build -t rag-api --build-arg WITH_QDRANT=true .
# Run (compose):    docker compose up --build -d

FROM python:3.12-slim

# Optional OCR support for scanned PDFs / image files: Tesseract for text
# recognition, Poppler for PDF -> image rendering.
ARG WITH_OCR=false
RUN if [ "$WITH_OCR" = "true" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Optional Qdrant backend (vector_store.provider: qdrant). Only the client is
# needed in the image; run Qdrant itself as the compose `qdrant` service or
# point qdrant_url at an external server.
ARG WITH_QDRANT=false

WORKDIR /app

# README.md is referenced by pyproject metadata, so it must be present.
COPY pyproject.toml README.md ./
COPY src ./src
RUN if [ "$WITH_QDRANT" = "true" ]; then \
        pip install --no-cache-dir ".[qdrant]"; \
    else \
        pip install --no-cache-dir .; \
    fi \
    && if [ "$WITH_OCR" = "true" ]; then \
        pip install --no-cache-dir pytesseract pdf2image Pillow; \
    fi

# Baked-in default config (server.host=0.0.0.0). docker-compose mounts the
# repo's config.docker.yaml over it so edits don't require a rebuild.
COPY config.docker.yaml ./config.yaml

# storage/ (Chroma + ingest index) and documents/ (corpus) are volumes.
RUN useradd --create-home raguser \
    && mkdir -p storage documents \
    && chown -R raguser:raguser /app
USER raguser

EXPOSE 8000

# python:slim has no curl; probe /health with the stdlib instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["rag-app", "serve", "--config", "config.yaml"]
