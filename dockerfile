# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by gensim, numpy, scipy
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement file
COPY requirements.txt /app/

# Install all python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Expose port (HuggingFace uses 7860)
EXPOSE 7860

# Run your Flask API using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
