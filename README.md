# Changepoint Detection In The Presence Of Outliers

This repository contains an implementation of the R-FPOP changepoint detection approach for time series with outliers, with an interactive Streamlit app to explore behavior across losses and hyperparameters.

The project is based on:
Fearnhead, P., & Rigaill, G. (2019). Changepoint Detection in the Presence of Outliers. Journal of the American Statistical Association, 114(525), 169-183.

## About This Project

This project is part of the **"Mise en Production des Projets de Data Science"** (Putting Data Science Projects into Production) course at ENSAE in 2026, taught by [Lino Galiana](https://github.com/linogaliana) and Romain Avouac.

For more information about the course, visit: https://ensae-reproductibilite.github.io

## What This Project Includes

- Robust changepoint detection algorithms in Python (`src/`)
- A Streamlit app (`app.py`) to test the algorithm interactively
- Toy datasets in `data/`
- Docker image support for local and cloud usage
- Kubernetes manifests in `deployment/` for cluster deployment

## Repository Layout

- `app.py`: Streamlit application entrypoint
- `src/`: core algorithm, losses, model selection, and visualization
- `data/`: built-in CSV examples used by the app
- `deployment/`: Kubernetes Deployment, Service, and Ingress manifests
- `scripts/run_docker.sh`: build and run Docker image locally
- `scripts/run_deployment.sh`: deploy to Kubernetes and port-forward locally

## Quick Start For Developers

### Prerequisites

- Python 3.x
- `pip`
- Optional: Docker
- Optional: `kubectl` with a configured cluster context

### Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run The App Locally

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Open: `http://localhost:8501`

## Run With Docker

### Option 1: Build Locally

```bash
docker build -t rfpop-streamlit-app .
docker run --rm -p 8501:8501 rfpop-streamlit-app
```

### Option 2: Use Helper Script

```bash
chmod +x scripts/run_docker.sh
scripts/run_docker.sh
```

Custom host port example:

```bash
scripts/run_docker.sh 8504
```

### Option 3: Pull From Docker Hub

```bash
docker pull audricms/r-fpop-change-points-detection:latest
docker run --rm -p 8501:8501 audricms/r-fpop-change-points-detection:latest
```

On Apple Silicon, if needed:

```bash
docker run --platform linux/amd64 --rm -p 8501:8501 audricms/r-fpop-change-points-detection:latest
```

## Run On Kubernetes (Current Setup)

This project deploys the Streamlit app using the manifests in `deployment/`.

### Kubernetes Files

- `deployment/deployment.yaml`
- `deployment/service.yaml`
- `deployment/ingress.yaml`

### Deploy

```bash
kubectl apply -f deployment/
kubectl get pods -l app=rfpop-app
kubectl logs -l app=rfpop-app -f --tail=200
```

### Helper Script (Deploy + Rollout + Logs + Port-Forward)

```bash
chmod +x scripts/run_deployment.sh
scripts/run_deployment.sh
```

Custom local port example:

```bash
scripts/run_deployment.sh 8502
```

### Ingress

Current host in `deployment/ingress.yaml`:

- `rfpop-change-points.lab.sspcloud.fr`

If you use a different domain, update both `spec.tls.hosts` and `spec.rules.host` in `deployment/ingress.yaml` before applying.

### Clean Up

```bash
kubectl delete -f deployment/
```

## Data Sources

The app can load toy CSVs from:

- Local files in `data/`
- Public SSPCloud MinIO URL (default):
  `https://minio.lab.sspcloud.fr/asicard/MPPDS - Projet`

Default toy filenames:

- `data example 1.csv`
- `data example 2.csv`
- `data example 3.csv`
- `data example 4.csv`

You can override the remote base URL with:

- `PUBLIC_DATA_URL`

## Development Commands

Install tools from `requirements.txt`, then use:

```bash
pytest
ruff check .
black .
pylint src app.py
```

Optional pre-commit setup:

```bash
pip install pre-commit
pre-commit install
```
