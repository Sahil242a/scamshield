# ScamShield — public demo image (Hugging Face Spaces, Render, Railway, Fly.io, any Docker host)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SCAMSHIELD_PUBLIC=1 SCAMSHIELD_RATE_LIMIT=30 PORT=7860

# Hugging Face Spaces runs containers as uid 1000; a non-root user is good practice everywhere
RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app . .
USER app

# Train inside the image: the model is always built by the exact library versions that will load it,
# and the image is reproducible from the repo alone (seeds are fixed). Takes ~2 minutes at build time.
RUN python data/build_dataset.py && python train.py && python -m pytest -q

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
# Most hosts inject $PORT; Hugging Face Spaces uses 7860
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
