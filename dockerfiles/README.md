# Dockerfiles Directory

Store Docker configurations and container setup files here.

## Contents

- **`Dockerfile`** — Main image definition for the application
- **`docker-compose.yaml`** — Multi-container orchestration (in project root)
- Docker-related configs and build scripts

## Usage

```bash
# Build image
docker build -f dockerfiles/Dockerfile -t mlops_se489:latest .

# Run container
docker compose up
```

## Phase

Phase 2 deliverable — Docker infrastructure setup.
