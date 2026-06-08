# PHASE 3: Continuous Machine Learning (CML) & Deployment

## Overview
Phase 3 implements continuous integration/continuous deployment (CI/CD) pipelines and productionizes Retail Demand Forecasting on cloud infrastructure. This phase covers automated testing, containerized workflows, CML integration, and multi-platform deployment options including GCP, Cloud Run, and serverless functions.

---

## 1. Continuous Integration & Testing

- [ ] **Unit Tests**: Write pytest test scripts for data processing and model components
- [ ] **Integration Tests**: Create integration tests for full training pipeline
- [ ] **Test Coverage**: Aim for >80% code coverage with pytest-cov
- [ ] **GitHub Actions - Tests**: Create workflow for running tests on every push
  - [ ] Trigger on: push to main/develop branches and PRs
  - [ ] Test across multiple Python versions if applicable
  - [ ] Report coverage metrics
- [ ] **GitHub Actions - Code Quality**: Create workflow for:
  - [ ] Running ruff linter
  - [ ] Type checking with mypy
  - [ ] Formatting checks
- [ ] **GitHub Actions - Docker Build**: Create workflow for building Docker image
  - [ ] Build on PR and main branch push
  - [ ] Test built image
- [ ] **Pre-commit Hooks**: Set up pre-commit hooks for:
  - [ ] Formatting (black/ruff)
  - [ ] Linting
  - [ ] Type checking
  - [ ] Trailing whitespace
- [ ] **Test Documentation**: Document how to run tests locally and in CI

---

## 2. Continuous Docker Building & CML

- [x] **Automated Docker Builds**: Configure Docker build pipeline triggered by:
  - [x] Commits to main branch
  - [x] Version tags
  - [x] Manual workflow dispatch
- [x] **Docker Push**: Implement push to container registry (Docker Hub, GitHub Container Registry, or GCP)
![alt text](image.png)
![alt text](gcp-artifact-registry-1.png)
![alt text](gcp-artifact-registry-2.png)
- [ ] **CML Initialization**: Initialize CML in repository
- [ ] **CML Workflow**: Create GitHub Actions workflow for CML that:
  - [ ] Trains model on workflow runner
  - [ ] Generates performance metrics
  - [ ] Creates visualizations/plots
  - [ ] Comments results on PR
- [ ] **CML Metrics Output**: Document format and sample output of CML metrics
- [ ] **CML Plots**: Generate sample plots and document in CML workflow
- [ ] **Model Comparison**: Create CML output showing comparison of current vs. baseline model
- [ ] **Workflow Documentation**: Document CML workflow setup and customization

---

## 3. Deployment on GCP

- [x] **GCP Project Setup**: Create GCP project and enable necessary APIs
- [x] **Service Account**: Create service account with appropriate permissions for:
  - [x] Artifact Registry
  - [x] Vertex AI
  - [x] Cloud Run
  - [x] Cloud Functions
  - [x] Compute Engine

- [x] **Artifact Registry**: Set up Artifact Registry for storing Docker images
  - [x] Create repository in Artifact Registry
  - [x] Configure authentication from CI/CD
  - [x] Push Docker images to registry
  ![alt text](gcp-artifact-registry-1.png)
  ![alt text](gcp-artifact-registry-2.png)
- [x] **Vertex AI Training (Option A)**: Set up custom training on Vertex AI
  - [x] Create training container image
  - [x] Configure training job specification
  - [x] Document how to submit training jobs
  ![alt text](gcp-job-model-bucket-landing.png)

First setup a bucket to store your data and trained model.
Navigate to GCP Console homepage -> Cloud Storage -> Create
Set a name for your bucket, location should be us-central1, storage class standard, access control uniform

Verify the bucket exists
gcloud storage ls

push your data to the bucket using the google cloud site interface

You can submit a job with the following command:
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=mlops489-train \
    --config=config_cpu.yaml \
    --service-account=trainer-sa@<project-id>.iam.gserviceaccount.com

To Watch the job, you stream the jogs directly to your terminal, and you can also check the logs directly on the google cloud

gcloud ai custom-jobs stream-logs <job-id> --region=us-central1

Once the job is finished, it should show up as green as shown in the following screenshots. The trained model is stored in the cloud storage bucket
  ![alt text](gcp-training-job-done-1.png)
  ![alt text](gcp-training-job-done-2.png)

Cleaning up all jobs with the following commands:

Find any running jobs
gcloud ai custom-jobs list --region=us-central1 \
    --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING"

Cancel the running job
gcloud ai custom-jobs cancel <job-id> --region=us-central1

Delete all instances
gcloud compute instances list
gcloud compute instances delete mlops489-train --zone=us-central1-a --quiet

- [ ] **Compute Engine Training (Option B)**: Set up training on Compute Engine instance
  - [ ] Create VM instance with GPU if needed
  - [ ] Document SSH access and training process
  - [ ] Set up instance for automated training
- [x] **Model Registry**: Store trained models in GCS bucket with versioning
  - [x] Create GCS bucket for models
  - [x] Implement model upload from training
  - [x] Document model retrieval process
  ![alt text](gcp-training-job-done-3.png)
- [ ] **FastAPI Service**: Create FastAPI application for model serving
  - [ ] Define inference endpoint(s)
  - [ ] Implement request validation
  - [ ] Add health check endpoint
  - [ ] Document API specification
- [ ] **Cloud Functions Deployment (Option A)**: Deploy inference as Cloud Function
  - [ ] Package model and FastAPI app for Cloud Functions
  - [ ] Create Cloud Function with appropriate memory/timeout
  - [ ] Configure HTTP trigger
  - [ ] Document invocation and response format
- [ ] **Cloud Run Deployment (Option B)**: Deploy as containerized service on Cloud Run
  - [ ] Create Dockerfile optimized for Cloud Run
  - [ ] Test locally with Cloud Run emulator
  - [ ] Deploy to Cloud Run with auto-scaling
  - [ ] Document deployment process
- [ ] **Streamlit/Gradio Deployment (Option C)**: Deploy demo app on HuggingFace Spaces
  - [ ] Create Streamlit or Gradio interface for model
  - [ ] Push to GitHub repository
  - [ ] Deploy to HuggingFace Spaces
  - [ ] Document feature walkthrough
- [ ] **Load Testing**: Test deployment with load testing tool (locust, Apache JMeter)
  - [ ] Establish baseline performance metrics
  - [ ] Document scaling characteristics
- [ ] **Monitoring Setup**: Configure Cloud Monitoring and Cloud Logging
  - [ ] Set up log aggregation
  - [ ] Create monitoring dashboards
  - [ ] Set up alerts for anomalies

---

## 4. Documentation & Repository Updates

- [ ] **Comprehensive README**: Update README with:
  - [ ] Architecture diagram showing all components
  - [ ] CI/CD pipeline overview
  - [ ] Deployment instructions for each option (Cloud Run, Cloud Functions, HuggingFace)
  - [ ] GCP setup and configuration guide
  - [ ] How to invoke deployed models
  - [ ] Monitoring and troubleshooting guide
  - [ ] Cost estimation and optimization tips
- [ ] **Deployment Guide**: Create detailed DEPLOYMENT.md with:
  - [ ] Step-by-step GCP setup instructions
  - [ ] Cloud Run deployment procedure
  - [ ] Cloud Functions configuration
  - [ ] Environment variables and secrets management
  - [ ] Rollback procedures
- [ ] **API Documentation**: Document all endpoints with:
  - [ ] Request/response schemas
  - [ ] Example curl/Python requests
  - [ ] Error codes and messages
- [ ] **Architecture Documentation**: Include diagrams showing:
  - [ ] Data pipeline
  - [ ] Training pipeline
  - [ ] Inference/serving architecture
  - [ ] CI/CD workflow
- [ ] **Screenshots/Demos**: Add:
  - [ ] Cloud Run dashboard screenshot
  - [ ] Monitoring dashboard screenshot
  - [ ] Streamlit/Gradio app screenshot
  - [ ] API response example
  - [ ] CML workflow output sample
- [ ] **Troubleshooting Guide**: Document solutions for:
  - [ ] Common deployment errors
  - [ ] Authentication issues
  - [ ] Performance problems
  - [ ] Cost overruns
- [ ] **Resource Cleanup Reminder**: Create CLEANUP.md with instructions for:
  - [ ] Deleting GCP resources (VMs, databases, etc.)
  - [ ] Cleaning up Cloud Storage buckets
  - [ ] Disabling APIs to avoid charges
  - [ ] Cost monitoring recommendations
- [ ] **Contributing Guide Update**: Update CONTRIBUTING.md with:
  - [ ] CI/CD requirements
  - [ ] Testing requirements for PRs
  - [ ] Deployment process documentation
- [ ] **Changelog**: Maintain CHANGELOG.md documenting releases and deployments

---

> **Checklist:** Use this as a guide for documenting your Phase 3 deliverables.
