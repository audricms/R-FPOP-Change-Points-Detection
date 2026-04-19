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
- Kubernetes manifests for dev testing in `deployment_dev/`
- GitOps-based deployment via a dedicated [application-deployment](https://github.com/vincentgraillat/application-deployment) repository and ArgoCD

## Repository Layout

- `app.py`: Streamlit application entry point
- `src/`: Core algorithm, loss functions, model selection, and visualization
- `data/`: Built-in CSV examples used by the app
- `.github/workflows/`: CI/CD pipelines
- `deployment_dev/`: Kubernetes Deployment, Service, Ingress, and Kustomization manifests
---

## For Developers

### Prerequisites

- Python 3.x
- `pip`
- Docker
- `kubectl` with a configured cluster context (e.g., SSPCloud)

### Local Setup

#### Dependencies

To isolate the project dependencies, it is recommended to use a virtual environment.
- To create the virtual environment: `python -m venv venv`
- To activate it: `source venv/bin/activate`
- To install the required dependencies: `pip install -r requirements.txt`

#### Pre-commit

Pre-commit automatically formats your code before each commit, ensuring that all developers follow the same formatting rules. To install it:
- Install Pre-commit: `pip install pre-commit`
- Set up Pre-commit in your project: `pre-commit install`
Once installed, Pre-commit will automatically run the defined checks and formatting before each commit.

#### Environment Variables

Create a `.env` file at the project root with the following variable:

```
S3_BUCKET="asicard"
S3_PREFIX="MPPDS - Projet"
```

This URL points to the public S3 bucket used to load toy datasets. The app will fall back to local files in `data/` if this variable is not set or the remote is unreachable.

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

## Run On Kubernetes (SSPCloud)

While developing, you can test your code in the cloud using Kubernetes. We use Kustomize so you can easily deploy the app to your personal Kubernetes namespace without altering the core deployment files.

### Configure Your Environment

Before deploying, open the `deployment_dev/kustomization.yaml` file and update it with your personal SSPCloud username:
1. Change `namespace: user-vgraillat` to your active namespace.
2. Update the two Ingress URL hostnames under the `patches:` section to ensure they match your personal URLs.

### Deploy

Once configured, deploy the application using the `-k` (Kustomize) flag:
```bash
kubectl apply -k deployment_dev/
```

### Monitor

Check the status of your pods and view logs:
```bash
kubectl get pods -l app=rfpop-app
kubectl logs -l app=rfpop-app -f --tail=200
```

### Clean Up

To remove the application from your cluster:
```bash
kubectl delete -k deployment_dev/
```

---

## Production Deployment (GitOps via ArgoCD)

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

---

## Streamlit app

The application is reachable at: `https://rfpop-vgraillat.user.lab.sspcloud.fr`
