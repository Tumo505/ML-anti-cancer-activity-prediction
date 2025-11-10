# Full Dockerfile with all data included for complete functionality
# This creates a ~2GB image but includes all data for dropdowns
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY pipeline.py .

# Copy model files
COPY saved_model/ ./saved_model/

# Copy ALL data files (includes full database for cell lines and drugs)
COPY data/ ./data/

# Copy sample files
COPY sample*.csv ./

# Expose port
EXPOSE 7860

# Set environment variable for port
ENV PORT=7860

# Run the application
CMD ["python", "app.py"]
