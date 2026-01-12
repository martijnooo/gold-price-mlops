FROM python:3.11-slim
WORKDIR /app

# Install dependencies
COPY streamlit_app/requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY streamlit_app/ /app/streamlit_app/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the app
WORKDIR /app/streamlit_app
CMD ["python", "gradio_app.py"]
