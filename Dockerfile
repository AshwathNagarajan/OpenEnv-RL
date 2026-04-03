FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install pydantic openenv-core openai

CMD ["python", "inference.py"]