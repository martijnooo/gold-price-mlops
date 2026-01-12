FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY api/requirements.txt .
RUN pip install -r requirements.txt

# Copy necessary source code
COPY api/ /app/api/
COPY data_pipeline/ /app/data_pipeline/
COPY features/ /app/features/

# Set working directory to api for running uvicorn
WORKDIR /app/api
ENV PYTHONPATH=/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
