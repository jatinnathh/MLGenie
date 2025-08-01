# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/data /app/models

# Initialize tracking files if they don't exist
RUN if [ ! -f global_dashboard_stats.json ]; then \
    echo '{"datasets_count":0,"models_trained":0,"best_accuracy":0.0,"total_training_time":0,"users_count":1,"ml_models_trained":0,"dl_models_trained":0,"total_predictions":0,"last_updated":"2025-08-01T15:35:00.000000"}' > global_dashboard_stats.json; \
    fi && \
    if [ ! -f global_activities.json ]; then \
    echo '[{"activity_type":"system_startup","description":"MLGenie dashboard tracking system initialized","status":"success","timestamp":"2025-08-01T15:35:00.000000","metadata":{}}]' > global_activities.json; \
    fi

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
