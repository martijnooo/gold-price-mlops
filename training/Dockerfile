FROM python:3.11-slim
WORKDIR /app
COPY training/requirements.txt .
RUN pip install -r requirements.txt
COPY training/ /app/training/
ENV PYTHONPATH=/app
CMD ["python", "training/training.py"]
