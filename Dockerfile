# Builder stage
FROM python:3.11-slim as builder

WORKDIR /opt/spam-phone

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /opt/spam-phone

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

EXPOSE 8000

CMD ["python", "-m", "scripts.run_api"]
