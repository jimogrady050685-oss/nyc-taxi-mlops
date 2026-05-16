# NYC Taxi Fare Prediction — MLOps Pipeline

ATU Donegal — Big Data Analytics & AI | Final MLOps Assignment (CA3, 40%)
Jim O'Grady | L00203776 | Due 24 May 2026

## Use Case

Operationalise a Gradient Boosted Trees fare prediction model (originally built on Databricks/PySpark for NYC TLC Yellow Taxi January 2025 data) as a containerised Flask API, deployed to a Kind Kubernetes cluster via Argo CD, with the full pipeline orchestrated by GitHub Actions.

## MLOps Stages Covered

| Stage | Implementation |
|-------|----------------|
| Data Acquisition & Preprocessing | `preprocess.py` runs in the `preprocess` job of `ml-pipeline.yml`; outputs are passed downstream as a GitHub artifact |
| Model Training & Testing | `train.py` runs in the `train` job; MLflow logs hyperparameters, metrics, and a versioned model artifact; `mlruns/` is uploaded as a workflow artifact for model lineage |
| Model Deployment | `Dockerfile` bakes the trained model into a single image; `build-image` job pushes to Docker Hub |
| Continuous Delivery | Argo CD watches the `k8s/` folder of this repo and auto-syncs `deployment.yaml` and `service.yaml` into the Kind cluster |
| Continuous Training | `retrain.yml` workflow (scheduled + manual + repository_dispatch) re-runs the entire pipeline and pushes a new model image |

## Architecture

```
GitHub Repo                             GitHub Actions
─────────────                           ──────────────
  preprocess.py  ────►  ml-pipeline.yml ────►  preprocess job (Ubuntu runner)
  train.py                                       │  features.parquet artifact
  app.py                                         ▼
  Dockerfile                                  train job (Ubuntu runner) ──► MLflow
  k8s/deployment.yaml                            │  model.joblib + mlruns artifacts
  k8s/service.yaml                               ▼
  argocd/application.yaml                     build-image job
                                                 │  docker push
                                                 ▼
                                         Docker Hub ◄────────────┐
                                                                 │
                                                                 │ image pull
                                                                 │
  k8s/*.yaml ─────────────────────────────► Argo CD ──► Kind Kubernetes Cluster
            (watched continuously)              │            │  taxi-fare pod
                                                │            │  taxi-fare-service
                                                └────────────┘  (NodePort 30500)
```

## Quick Start

1. Fork this repo and set the GitHub secrets:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
2. Edit `k8s/deployment.yaml` — replace `YOUR_DOCKERHUB_USERNAME`
3. Edit `argocd/application.yaml` — replace `YOUR_GITHUB_USERNAME`
4. On any Docker-enabled Linux box: `./scripts/setup_kind.sh`
5. Push a change to `main` to trigger the first pipeline run
6. Watch Argo CD auto-deploy the new image to the cluster

## Branching Strategy

Simplified GitFlow: `main` (production) ← `develop` (integration) ← `feature/*` (development).

See `docs/branching.md` for details.
