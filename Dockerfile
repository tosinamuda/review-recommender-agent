FROM registry.access.redhat.com/ubi9/python-312:latest

# ── System-level setup (root) ─────────────────────────────────────────────────
USER root

# Install uv to a system-wide path via the official install script
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /opt/app-root/src

# ── Python dependencies (cached layer — rebuilt only when lockfile changes) ───
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# ── Pre-download embedding model (avoids cold-start network fetch) ─────────────
RUN uv run python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('BAAI/bge-small-en-v1.5')"

# ── Application source and pre-built data artifacts ───────────────────────────
COPY src/ ./src/
COPY data/ ./data/

# Hand ownership to the default UBI non-root user (uid 1001, gid 0)
RUN chown -R 1001:0 /opt/app-root/src && chmod -R g=u /opt/app-root/src

# ── Drop privileges ────────────────────────────────────────────────────────────
USER 1001

EXPOSE 8000

# Bind to all interfaces inside the container so -p 8000:8000 works.
# Override at runtime with APP_HOST=127.0.0.1 if needed.
ENV APP_HOST=0.0.0.0

# Pass LM_MODEL + the matching provider API key at runtime:
#   docker run --env-file .env -p 8000:8000 bct-agent
CMD ["uv", "run", "app-start"]
