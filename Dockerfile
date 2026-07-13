# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create and activate virtual environment, then install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Run
FROM python:3.11-slim as final

WORKDIR /app

# Install system runtime requirements (e.g. libpq for pgsql, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

# Setup non-root execution user and assign ownership
RUN groupadd -r mnipgrp && useradd -r -g mnipgrp mnipuser
RUN mkdir -p /app/data /app/models /app/reports && chown -R mnipuser:mnipgrp /app

# Switch to the non-root user
USER mnipuser

# Copy application source code
COPY --chown=mnipuser:mnipgrp . .

# Expose API and Streamlit Dashboard ports
EXPOSE 8000
EXPOSE 8501

# Default command runs the API, overridden in docker-compose for UI
CMD ["uvicorn", "mnip.main:app", "--host", "0.0.0.0", "--port", "8000"]
