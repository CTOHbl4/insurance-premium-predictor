FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# Create parent checkpoint directory
RUN mkdir -p /app/ml/checkpoints

CMD ["python", "services/trainer/trainer.py"]