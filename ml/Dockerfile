FROM python:3.11-slim

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY ./ml/README.md ./ml/pyproject.toml ./ml/poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root
 
COPY ./ml/ .

EXPOSE 5000

CMD ["mlflow", "ui", "--host", "0.0.0.0"]