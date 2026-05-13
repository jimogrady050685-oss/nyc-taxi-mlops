# NYC Taxi Fare MLOps Pipeline

**Module:** CA3 Big Data Analytics & AI
**Author:** Jim O'Grady | L00203776 | ATU Donegal
**Due:** 24 May 2026

## Overview

MLOps pipeline that operationalises a Gradient Boosted Trees fare prediction model for NYC Yellow Taxi trips. Migrates the original Databricks/PySpark implementation to scikit-learn, packages it as a Docker container, and deploys it to a GCP VM via GitHub Actions.

## MLOps Stages Implemented

1. Data Acquisition & Preprocessing (preprocess.py)
2. Model Training & Testing (train.py)
3. Model Deployment (app.py + Flask + Docker)
4. Continuous Integration (.github/workflows/ci.yml)
5. Continuous Delivery (.github/workflows/cd.yml)
6. Continuous Training (.github/workflows/ct.yml)

## Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **API:** Flask + gunicorn
- **Container:** Docker, Docker Hub
- **CI/CD:** GitHub Actions
- **Deployment:** Google Cloud Platform (Compute Engine VM)

## Status

Skeleton structure in place. Implementation in progress.
