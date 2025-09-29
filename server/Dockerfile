FROM python:3.11-slim

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY ./server/README.md ./server/pyproject.toml ./server/poetry.lock ./

# This is a path dependency
COPY ./ml/ml ./ml

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root



COPY ./server/server ./server

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]