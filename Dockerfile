FROM python:3.10-slim

WORKDIR /app

# Install FastAPI and Uvicorn from PyPI
RUN pip install --no-cache-dir fastapi uvicorn

# Install PyTorch CPU from official PyTorch index
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy API code
COPY api.py .

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
