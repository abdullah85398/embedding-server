# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
ARG INSTALL_TYPE=cpu
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$INSTALL_TYPE" = "cpu" ]; then \
        echo "Installing CPU-only PyTorch"; \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Installing Standard PyTorch (CUDA)"; \
        pip install --no-cache-dir torch; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download models
# We need to copy the script and config to the builder stage
COPY models.yaml .
COPY download_models.py .
# Set HF_HOME to a specific location we can copy later
ENV HF_HOME=/app/hf_cache
RUN python download_models.py

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Define Environment Variables
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV LOG_LEVEL=INFO
ENV WORKERS=1
ENV AUTH_MODE=NONE

# Copy configuration and code
COPY models.yaml .
COPY main.py .
COPY app ./app

# Copy downloaded models from builder
# We copy them to the user's cache directory
COPY --from=builder /app/hf_cache /home/appuser/.cache/huggingface

# Change ownership
RUN chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Expose ports (HTTP and gRPC)
EXPOSE 8000
EXPOSE 50051

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]
