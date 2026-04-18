# Changepoint Detection in the Presence of Outliers

This repository contains an implementation of the R-FPOP changepoint detection approach for time series with outliers, packaged with an interactive Streamlit app to explore the algorithm's behavior across different loss functions and hyperparameters.

The project is based on:
> Fearnhead, P., & Rigaill, G. (2019). *Changepoint Detection in the Presence of Outliers*. Journal of the American Statistical Association, 114(525), 169-183.

## About This Project

This project is part of the **"Mise en Production des Projets de Data Science"** (Bringing Data Science to Production) course at ENSAE (2026), taught by [Lino Galiana](https://github.com/linogaliana) and Romain Avouac.

For more information about the course, visit: [ensae-reproductibilite.github.io](https://ensae-reproductibilite.github.io)

## What This Project Includes

- Robust changepoint detection algorithms implemented in Python (`src/`)
- A Streamlit web application (`app.py`) to test the algorithm interactively
- Toy datasets located in `data/`
- Docker image support for both local execution and cloud deployment
- CI/CD pipelines via GitHub Actions (linting, tests, Docker Hub build & push)
- GitOps-based deployment via a dedicated [application-deployment](https://github.com/vincentgraillat/application-deployment) repository and ArgoCD

## Repository Layout

- `app.py`: Streamlit application entry point
- `src/`: Core algorithm, loss functions, model selection, and visualization
- `data/`: Built-in CSV examples used by the app
- `.github/workflows/`: CI/CD pipelines (code quality, tests, Docker Hub build & push)

---

## Quick Start For Developers

### Prerequisites

- Python 3.x
- `pip`
- Optional: Docker
- Optional: `kubectl` with a configured cluster context (e.g., SSPCloud)

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
Open your browser and navigate to: `http://localhost:8501`

---

## Run With Docker

### Build & Run Locally

```bash
docker build -t rfpop-streamlit-app .
docker run --rm -p 8501:8501 rfpop-streamlit-app
```

### Docker Hub Usage

Docker Hub is the image registry between the CI pipeline and the Kubernetes cluster. The `prod.yml` workflow triggers on two events:

- **Push to `main`**: builds and pushes an image tagged `main`.
- **Git tag `v*.*.*`**: builds and pushes an image tagged with the exact version (e.g., `v1.2.0`). This is the tag used for production deployments.

Image tagging is handled automatically by `docker/metadata-action`. You never need to push to Docker Hub manually.

---

## Deployment (GitOps via ArgoCD)

Kubernetes deployment is managed by a separate, dedicated repository following GitOps principles: **[application-deployment](https://github.com/vincentgraillat/application-deployment)**.

### How it works

```
Create a git tag vX.Y.Z on this repo
        │
        ▼
GitHub Actions builds & pushes audricms/r-fpop-change-points-detection:vX.Y.Z to Docker Hub
        │
Update image tag in application-deployment/deployment/deployment.yaml
        │
        ▼
ArgoCD detects manifest drift → auto-syncs cluster → new image is live
```

**To release a new version:**
1. Push a tag to this repo: `git tag vX.Y.Z && git push origin vX.Y.Z`
2. Wait for the GitHub Actions build to complete.
3. Update the image tag in `application-deployment/deployment/deployment.yaml` to `vX.Y.Z` and merge to `main`.
4. ArgoCD detects the change and rolls out the new image automatically.

---

## Data Sources

The application loads toy CSV datasets from two potential sources:

1. **Local files** stored in the `data/` directory.
2. **Public S3 Storage** (SSPCloud MinIO).

The default remote base URL is configured as:
`https://minio.lab.sspcloud.fr/asicard/MPPDS%20-%20Projet`

You can override this remote base URL by setting the following environment variable in your deployment configuration:
- `PUBLIC_DATA_URL`

## Development Commands

Ensure you have installed the development tools from `requirements.txt`. You can run the following checks:

```bash
pytest
ruff check .
black .
pylint src app.py
```

It is highly recommended to use `pre-commit` to automate formatting before pushing code:

```bash
pip install pre-commit
pre-commit install
```
