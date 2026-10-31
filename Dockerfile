FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    EMBEDDER_DEVICE=cpu \
    HF_HOME=/home/app/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app/backend
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=app:app backend/ /app/backend/
COPY --from=frontend --chown=app:app /build/dist/ /app/frontend/dist/
RUN mkdir -p /app/backend/data/indexes /app/backend/data/prepared \
    /app/backend/data/cache /app/snomed /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache
USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
