# Dockerfile for stat_arb_prod
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Expose port for web services
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
# DEPRECATED: statistics_service has been replaced by analytics.stats
# CMD ["gunicorn", "statistics_service.app:app", "--bind", "0.0.0.0:8000"]
CMD ["python", "-m", "analytics.example_usage"]
