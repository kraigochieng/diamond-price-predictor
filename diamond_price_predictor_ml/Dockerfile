FROM python:3.11-slim

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY README.md pyproject.toml poetry.lock /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root
 
COPY . /app/

EXPOSE 5000

CMD ["mlflow", "ui", "--host", "0.0.0.0"]