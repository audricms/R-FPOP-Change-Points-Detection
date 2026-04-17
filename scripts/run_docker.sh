#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-rfpop-streamlit-app}"
CONTAINER_NAME="${CONTAINER_NAME:-rfpop-streamlit-app}"
HOST_PORT="${1:-8501}"
CONTAINER_PORT="8501"

if [[ "${HOST_PORT}" =~ ^-h|--help$ ]]; then
  echo "Usage: scripts/run_docker.sh [HOST_PORT]"
  echo "Example: scripts/run_docker.sh 8504"
  exit 0
fi

if ! [[ "${HOST_PORT}" =~ ^[0-9]+$ ]]; then
  echo "Error: HOST_PORT must be a number." >&2
  exit 1
fi

echo "Building image ${IMAGE_NAME}..."
docker build -t "${IMAGE_NAME}" .

# Remove any previous container with the same name.
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting container ${CONTAINER_NAME} on http://localhost:${HOST_PORT} ..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  "${IMAGE_NAME}" \
  streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=${CONTAINER_PORT} \
  --server.headless=true \
  --browser.serverAddress=localhost \
  --browser.serverPort=${HOST_PORT}

echo "Container started."
echo "Open: http://localhost:${HOST_PORT}"
echo "Follow logs: docker logs -f ${CONTAINER_NAME}"
echo "Stop: docker rm -f ${CONTAINER_NAME}"
