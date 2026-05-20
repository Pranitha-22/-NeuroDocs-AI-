FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NEURODOCS_MAX_UPLOAD_MB=50
ENV NEURODOCS_RATE_LIMIT_PER_MINUTE=60

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --shell /bin/bash neurodocs \
    && mkdir -p /app/data \
    && chown -R neurodocs:neurodocs /app

COPY --chown=neurodocs:neurodocs . .

USER neurodocs

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
