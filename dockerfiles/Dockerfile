
# Multi-stage build for Retail Demand Forecasting

# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

# Stage 2: Runtime
FROM builder AS runtime

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy project code
COPY . .

# Install the package itself without reinstalling dependencies
RUN pip install --user --no-deps -e .

EXPOSE 8000

ENTRYPOINT ["sh", "-u", "pipeline.sh"]
