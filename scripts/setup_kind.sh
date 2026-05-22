#!/bin/bash
# scripts/setup_kind.sh
# ======================
# Run this once on the machine that will host the Kind cluster.
# Works on any Linux machine that has Docker available.
#
# What it does:
#   Installs kind and kubectl if missing, creates a Kind cluster with the
#   correct port mapping, installs Argo CD into it, and applies the
#   application manifest that points Argo CD at this GitHub repo.
#   After this runs, the cluster manages itself from Git.
#
# Prerequisites:
#   Docker installed and running
#   kubectl installed  https://kubernetes.io/docs/tasks/tools/
#   kind installed     https://kind.sigs.k8s.io/docs/user/quick-start/
#
# Usage:
#   chmod +x scripts/setup_kind.sh
#   ./scripts/setup_kind.sh

set -e

# Install kind if it is not already on the machine
if ! command -v kind &> /dev/null; then
    echo "Installing kind..."
    curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-amd64
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
fi

# Install kubectl if it is not already on the machine
if ! command -v kubectl &> /dev/null; then
    echo "Installing kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x ./kubectl
    sudo mv ./kubectl /usr/local/bin/kubectl
fi

# Create the Kind cluster.
# The extraPortMappings block maps port 30500 on the host to port 30500
# on the cluster node so the Flask API is reachable via curl from outside.
echo "=== Creating Kind cluster ==="
cat <<EOF > /tmp/kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: argocd-demo
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30500
        hostPort:      30500
        protocol:      TCP
EOF

kind create cluster --config /tmp/kind-config.yaml || echo "Cluster already exists, continuing..."

kubectl cluster-info --context kind-argocd-demo

# Install Argo CD into its own namespace inside the cluster
echo "=== Installing Argo CD ==="
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for Argo CD to finish coming up before continuing
echo "Waiting for Argo CD pods to be ready..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/argocd-server -n argocd

echo ""
echo "=== Argo CD is ready ==="
echo ""

# Print the initial admin password so you can log into the UI
ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret \
    -o jsonpath="{.data.password}" | base64 -d)
echo "Argo CD admin password (username is admin):"
echo "  $ARGOCD_PASSWORD"
echo ""

# Instructions for accessing the UI
echo "To open the Argo CD UI, run this in a separate terminal:"
echo "  kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo ""
echo "Then go to https://localhost:8080 and accept the self-signed cert."
echo ""

# Apply the application manifest that tells Argo CD what to watch and where to deploy it
echo "=== Applying Argo CD Application manifest ==="
kubectl apply -f argocd/application.yaml

echo ""
echo "Setup complete."
echo ""
echo "Argo CD is now watching the k8s/ folder of the GitHub repo."
echo "Within about 3 minutes it will deploy the taxi-fare application to the default namespace."
echo ""
echo "To watch it come up:"
echo "  kubectl get pods -w"
echo ""
echo "To test the API once it is running:"
echo "  curl http://localhost:30500/health"