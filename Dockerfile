FROM node:22.12.0-bookworm-slim@sha256:35531c52ce27b6575d69755c73e65d4468dba93a25644eed56dc12879cae9213 AS frontend

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp/app-home \
    HF_HOME=/tmp/huggingface

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /srv/app

COPY requirements.txt ml/requirements-inference.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 \
    && pip install --no-cache-dir --requirement requirements-inference.txt

COPY app ./app
COPY ingestion ./ingestion
COPY ml ./ml
COPY knowledge ./knowledge
COPY artifacts/models/legal-bert-cuad ./artifacts/models/legal-bert-cuad
COPY artifacts/models/selected.json ./artifacts/models/selected.json
COPY artifacts/knowledge/federal-v2.sqlite ./artifacts/knowledge/federal-v2.sqlite
COPY scripts/verify-shared-artifacts.py ./scripts/verify-shared-artifacts.py
COPY --from=frontend /build/frontend/dist ./frontend/dist
COPY docker-entrypoint.py /usr/local/bin/app-entrypoint
RUN python scripts/verify-shared-artifacts.py \
    && chown -R app:app /srv/app \
    && chmod 0555 /usr/local/bin/app-entrypoint

USER app
ENTRYPOINT ["python", "/usr/local/bin/app-entrypoint"]

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
