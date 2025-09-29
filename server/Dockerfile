FROM python:3.11-slim

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY README.md pyproject.toml poetry.lock /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root



COPY server /app/server

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]