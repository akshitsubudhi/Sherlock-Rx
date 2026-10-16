FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests/data ./tests/data
RUN pip install -e ".[llm,ui]"
# Data and models are built into the image so containers start instantly and identically.
RUN sherlock generate --scale 1 --seed 7 && sherlock train
ENV SHERLOCK_DUCKDB_THREADS=2
EXPOSE 8000 8501
CMD ["uvicorn", "sherlock_rx.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
