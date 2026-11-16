FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEV_CACHE_DIR=/tmp/ai-lawyer-cache
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY --chown=10001:10001 *.py ./
COPY --chown=10001:10001 static ./static
RUN chown 10001:10001 /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=180s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
