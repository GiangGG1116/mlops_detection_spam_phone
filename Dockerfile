FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    # Create non-root user for security
    && useradd --create-home --shell /bin/bash appuser

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Ensure data / model directories exist and are writable by appuser
RUN mkdir -p data/runs data/predictions data/train models/production models/candidates models/archive \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Default: CLI mode. docker-compose overrides this with uvicorn for API mode.
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
