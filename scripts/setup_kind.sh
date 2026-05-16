#!/bin/bash
# scripts/setup_kind.sh — One-time Kind + Argo CD setup
# =======================================================
# Run this ONCE on the machine that will host the Kind cluster.
# The machine can be: your laptop, a GCP free-tier e2-micro VM, or anywhere
# Linux + Docker is available.
#
# This script follows the lecturer's tutorial step-by-step from 16 May.
#
# Prerequisites:
#   - Docker installed
#   - kubectl installed (https://kubernetes.io/docs/tasks/tools/)
#   - kind installed  (https://kind.sigs.k8s.io/docs/user/quick-start/)
#
# Usage:
#   chmod +x scripts/setup_kind.sh
#   ./scripts/setup_kind.sh
#
# Video talking point:
#   "setup_kind.sh creates the Kind cluster, installs Argo CD into it,
#    and applies my Argo CD application manifest pointing at my GitHub repo.
#    After this runs once, the cluster auto-pulls all deployments from
#    GitHub — I never need to kubectl apply anything by hand again."

set -e

# ── Step 1: Install kind if missing (Linux x86_64 instructions) ───────────────
if ! command -v kind &> /dev/null; then
    echo "Installing kind..."
    curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-amd64
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
fi

# ── Step 2: Install kubectl if missing ────────────────────────────────────────
if ! command -v kubectl &> /dev/null; then
    echo "Installing kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x ./kubectl
    sudo mv ./kubectl /usr/local/bin/kubectl
fi

# ── Step 3: Create the Kind cluster with a node-port mapping ──────────────────
# We map the cluster's nodePort (30500 from service.yaml) to the host's
# port 30500 so we can curl the API from outside the cluster.
echo "=== Creating Kind cluster ==="
cat <<EOF > /tmp/kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: argocd-demo
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: 30500    # node-side port (matches service.yaml nodePort)
        hostPort:      30500    # host-side port
        protocol:      TCP
EOF

kind create cluster --config /tmp/kind-config.yaml || echo "Cluster already exists, continuing..."

# Tell kubectl to talk to this cluster
kubectl cluster-info --context kind-argocd-demo

# ── Step 4: Install Argo CD into the cluster ─────────────────────────────────
echo "=== Installing Argo CD ==="
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for Argo CD pods to become ready (this takes a few minutes)
echo "Waiting for Argo CD pods to be ready..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/argocd-server -n argocd

echo ""
echo "=== Argo CD is ready ==="
echo ""

# ── Step 5: Print Argo CD initial admin password ─────────────────────────────
ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret \
    -o jsonpath="{.data.password}" | base64 -d)
echo "Argo CD admin password (username 'admin'):"
echo "  $ARGOCD_PASSWORD"
echo ""

# ── Step 6: Set up port-forward for the Argo CD UI ────────────────────────────
echo "To access the Argo CD UI, in another terminal run:"
echo "  kubectl port-forward svc/argocd-server -n argocd 8080:443"
echo ""
echo "Then browse to: https://localhost:8080  (accept the self-signed cert)"
echo ""

# ── Step 7: Apply the Argo CD Application manifest ───────────────────────────
echo "=== Applying Argo CD Application manifest ==="
echo "IMPORTANT: edit argocd/application.yaml first and replace"
echo "  YOUR_GITHUB_USERNAME with your actual GitHub username."
echo ""
read -p "Press ENTER after you have edited the file, or Ctrl-C to abort..."

kubectl apply -f argocd/application.yaml

echo ""
echo "✓ Setup complete!"
echo ""
echo "Argo CD is now monitoring your GitHub repo. Within ~3 minutes it will"
echo "automatically deploy the taxi-fare application to the default namespace."
echo ""
echo "To watch the deployment:"
echo "  kubectl get pods -w"
echo ""
echo "To test the API once it is running:"
echo "  curl http://localhost:30500/health"
