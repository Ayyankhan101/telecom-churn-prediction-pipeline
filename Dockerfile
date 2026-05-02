FROM python:3.13-slim

WORKDIR /app

# Cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source only
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1
ENV FLASK_DEBUG=false
ENV MODEL_PATH=/app/models

EXPOSE 5000 8501 8765

CMD ["python", "src/deployment/api/app.py"]