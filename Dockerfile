FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp/app-home \
    GRADIO_TEMP_DIR=/tmp/gradio \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/tmp/huggingface

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir --requirement requirements.txt

COPY app ./app
COPY ingestion ./ingestion
COPY ml ./ml
COPY knowledge ./knowledge
COPY docker-entrypoint.py /usr/local/bin/app-entrypoint
RUN chown -R app:app /srv/app && chmod 0555 /usr/local/bin/app-entrypoint

USER app
ENTRYPOINT ["python", "/usr/local/bin/app-entrypoint"]

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
