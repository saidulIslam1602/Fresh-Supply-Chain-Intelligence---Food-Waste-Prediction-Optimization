#!/bin/bash

# Fresh Supply Chain Intelligence System - Production Deployment Script
# Advanced deployment with blue-green strategy, health checks, and rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"
NAMESPACE="fresh-supply-chain"
APP_NAME="fresh-supply-chain-intelligence"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
VERSION="${VERSION:-$(git rev-parse --short HEAD)}"
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-rolling}"  # rolling, blue-green, canary

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed (optional)
    if command -v helm &> /dev/null; then
        log_info "Helm is available for advanced deployments"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "${PROJECT_ROOT}"
    
    # Build API image
    log_info "Building API image..."
    docker build -f deployment/docker/Dockerfile.api -t "${DOCKER_REGISTRY}/${APP_NAME}/api:${VERSION}" .
    docker tag "${DOCKER_REGISTRY}/${APP_NAME}/api:${VERSION}" "${DOCKER_REGISTRY}/${APP_NAME}/api:latest"
    
    # Build Dashboard image
    log_info "Building Dashboard image..."
    docker build -f deployment/docker/Dockerfile.dashboard -t "${DOCKER_REGISTRY}/${APP_NAME}/dashboard:${VERSION}" .
    docker tag "${DOCKER_REGISTRY}/${APP_NAME}/dashboard:${VERSION}" "${DOCKER_REGISTRY}/${APP_NAME}/dashboard:latest"
    
    log_success "Docker images built successfully"
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."
    
    docker push "${DOCKER_REGISTRY}/${APP_NAME}/api:${VERSION}"
    docker push "${DOCKER_REGISTRY}/${APP_NAME}/api:latest"
    docker push "${DOCKER_REGISTRY}/${APP_NAME}/dashboard:${VERSION}"
    docker push "${DOCKER_REGISTRY}/${APP_NAME}/dashboard:latest"
    
    log_success "Images pushed to registry"
}

# Create namespace and apply base resources
setup_namespace() {
    log_info "Setting up namespace and base resources..."
    
    cd "${DEPLOYMENT_DIR}/kubernetes"
    
    # Apply namespace
    kubectl apply -f namespace.yaml
    
    # Apply RBAC
    kubectl apply -f rbac.yaml
    
    # Apply ConfigMaps and Secrets
    kubectl apply -f configmap.yaml
    kubectl apply -f secrets.yaml
    
    # Apply Persistent Volumes
    kubectl apply -f persistent-volumes.yaml
    
    log_success "Namespace and base resources created"
}

# Deploy database and cache
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    cd "${DEPLOYMENT_DIR}/kubernetes"
    
    # Deploy database and cache
    kubectl apply -f database-deployment.yaml
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    kubectl wait --for=condition=ready pod -l app=sqlserver -n ${NAMESPACE} --timeout=300s
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n ${NAMESPACE} --timeout=120s
    
    log_success "Infrastructure components deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    cd "${DEPLOYMENT_DIR}/kubernetes"
    
    # Deploy monitoring components
    kubectl apply -f monitoring-deployment.yaml
    
    # Wait for monitoring to be ready
    log_info "Waiting for monitoring components to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n ${NAMESPACE} --timeout=180s
    kubectl wait --for=condition=ready pod -l app=grafana -n ${NAMESPACE} --timeout=180s
    kubectl wait --for=condition=ready pod -l app=jaeger -n ${NAMESPACE} --timeout=120s
    
    log_success "Monitoring stack deployed"
}

# Rolling deployment strategy
deploy_rolling() {
    log_info "Performing rolling deployment..."
    
    cd "${DEPLOYMENT_DIR}/kubernetes"
    
    # Update image tags in deployment files
    sed -i.bak "s|image: fresh-supply-chain/api:.*|image: ${DOCKER_REGISTRY}/${APP_NAME}/api:${VERSION}|g" api-deployment.yaml
    sed -i.bak "s|image: fresh-supply-chain/dashboard:.*|image: ${DOCKER_REGISTRY}/${APP_NAME}/dashboard:${VERSION}|g" dashboard-deployment.yaml
    
    # Apply deployments
    kubectl apply -f api-deployment.yaml
    kubectl apply -f dashboard-deployment.yaml
    
    # Wait for rollout to complete
    log_info "Waiting for API deployment rollout..."
    kubectl rollout status deployment/api-deployment -n ${NAMESPACE} --timeout=600s
    
    log_info "Waiting for Dashboard deployment rollout..."
    kubectl rollout status deployment/dashboard-deployment -n ${NAMESPACE} --timeout=300s
    
    # Restore original files
    mv api-deployment.yaml.bak api-deployment.yaml
    mv dashboard-deployment.yaml.bak dashboard-deployment.yaml
    
    log_success "Rolling deployment completed"
}

# Blue-green deployment strategy
deploy_blue_green() {
    log_info "Performing blue-green deployment..."
    
    # This is a simplified blue-green deployment
    # In production, you might want to use more sophisticated tools like Argo Rollouts
    
    cd "${DEPLOYMENT_DIR}/kubernetes"
    
    # Create green deployment
    sed "s/api-deployment/api-deployment-green/g; s|image: fresh-supply-chain/api:.*|image: ${DOCKER_REGISTRY}/${APP_NAME}/api:${VERSION}|g" api-deployment.yaml > api-deployment-green.yaml
    sed "s/dashboard-deployment/dashboard-deployment-green/g; s|image: fresh-supply-chain/dashboard:.*|image: ${DOCKER_REGISTRY}/${APP_NAME}/dashboard:${VERSION}|g" dashboard-deployment.yaml > dashboard-deployment-green.yaml
    
    # Deploy green version
    kubectl apply -f api-deployment-green.yaml
    kubectl apply -f dashboard-deployment-green.yaml
    
    # Wait for green deployment to be ready
    log_info "Waiting for green deployment to be ready..."
    kubectl wait --for=condition=available deployment/api-deployment-green -n ${NAMESPACE} --timeout=600s
    kubectl wait --for=condition=available deployment/dashboard-deployment-green -n ${NAMESPACE} --timeout=300s
    
    # Health check green deployment
    if health_check_green; then
        log_info "Green deployment health check passed. Switching traffic..."
        
        # Update service selectors to point to green deployment
        kubectl patch service api-service -n ${NAMESPACE} -p '{"spec":{"selector":{"version":"green"}}}'
        kubectl patch service dashboard-service -n ${NAMESPACE} -p '{"spec":{"selector":{"version":"green"}}}'
        
        # Wait a bit for traffic to switch
        sleep 30
        
        # Delete old blue deployment
        kubectl delete deployment api-deployment -n ${NAMESPACE} --ignore-not-found=true
        kubectl delete deployment dashboard-deployment -n ${NAMESPACE} --ignore-not-found=true
        
        # Rename green to blue for next deployment
        kubectl patch deployment api-deployment-green -n ${NAMESPACE} --type='merge' -p='{"metadata":{"name":"api-deployment"}}'
        kubectl patch deployment dashboard-deployment-green -n ${NAMESPACE} --type='merge' -p='{"metadata":{"name":"dashboard-deployment"}}'
        
        log_success "Blue-green deployment completed successfully"
    else
        log_error "Green deployment health check failed. Rolling back..."
        kubectl delete deployment api-deployment-green -n ${NAMESPACE}
        kubectl delete deployment dashboard-deployment-green -n ${NAMESPACE}
        exit 1
    fi
    
    # Cleanup
    rm -f api-deployment-green.yaml dashboard-deployment-green.yaml
}

# Health check for green deployment
health_check_green() {
    log_info "Performing health check on green deployment..."
    
    # Get green pod IPs
    API_POD=$(kubectl get pods -l app=api,version=green -n ${NAMESPACE} -o jsonpath='{.items[0].status.podIP}')
    DASHBOARD_POD=$(kubectl get pods -l app=dashboard,version=green -n ${NAMESPACE} -o jsonpath='{.items[0].status.podIP}')
    
    # Health check API
    if kubectl run health-check-api --rm -i --restart=Never --image=curlimages/curl -- curl -f "http://${API_POD}:8000/health" > /dev/null 2>&1; then
        log_info "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Health check Dashboard
    if kubectl run health-check-dashboard --rm -i --restart=Never --image=curlimages/curl -- curl -f "http://${DASHBOARD_POD}:8050/health" > /dev/null 2>&1; then
        log_info "Dashboard health check passed"
    else
        log_error "Dashboard health check failed"
        return 1
    fi
    
    return 0
}

# Deploy ingress and networking
deploy_networking() {
    log_info "Deploying ingress and networking..."
    
    cd "${DEPLOYMENT_DIR}/kubernetes"
    
    # Apply ingress
    kubectl apply -f ingress.yaml
    
    log_success "Networking deployed"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check all pods are running
    log_info "Checking pod status..."
    kubectl get pods -n ${NAMESPACE}
    
    # Check services
    log_info "Checking services..."
    kubectl get services -n ${NAMESPACE}
    
    # Check ingress
    log_info "Checking ingress..."
    kubectl get ingress -n ${NAMESPACE}
    
    # Health checks
    log_info "Performing health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=api -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=ready pod -l app=dashboard -n ${NAMESPACE} --timeout=300s
    
    log_success "Deployment verification completed"
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # Rollback API deployment
    kubectl rollout undo deployment/api-deployment -n ${NAMESPACE}
    kubectl rollout undo deployment/dashboard-deployment -n ${NAMESPACE}
    
    # Wait for rollback to complete
    kubectl rollout status deployment/api-deployment -n ${NAMESPACE} --timeout=300s
    kubectl rollout status deployment/dashboard-deployment -n ${NAMESPACE} --timeout=300s
    
    log_success "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    cd "${DEPLOYMENT_DIR}/kubernetes"
    rm -f *.bak *-green.yaml
}

# Main deployment function
main() {
    log_info "Starting deployment of Fresh Supply Chain Intelligence System"
    log_info "Version: ${VERSION}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Strategy: ${DEPLOYMENT_STRATEGY}"
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Check prerequisites
    check_prerequisites
    
    # Build and push images
    if [[ "${SKIP_BUILD:-false}" != "true" ]]; then
        build_images
        if [[ "${SKIP_PUSH:-false}" != "true" ]]; then
            push_images
        fi
    fi
    
    # Setup namespace and base resources
    setup_namespace
    
    # Deploy infrastructure
    deploy_infrastructure
    
    # Deploy monitoring
    deploy_monitoring
    
    # Deploy application based on strategy
    case ${DEPLOYMENT_STRATEGY} in
        "rolling")
            deploy_rolling
            ;;
        "blue-green")
            deploy_blue_green
            ;;
        *)
            log_error "Unknown deployment strategy: ${DEPLOYMENT_STRATEGY}"
            exit 1
            ;;
    esac
    
    # Deploy networking
    deploy_networking
    
    # Verify deployment
    verify_deployment
    
    log_success "Deployment completed successfully!"
    log_info "Access the application at:"
    log_info "  API: http://api.fresh-supply-chain.local"
    log_info "  Dashboard: http://dashboard.fresh-supply-chain.local"
    log_info "  Grafana: http://grafana.fresh-supply-chain.local"
    log_info "  Jaeger: http://jaeger.fresh-supply-chain.local"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        rollback_deployment
        ;;
    "verify")
        verify_deployment
        ;;
    "cleanup")
        log_info "Cleaning up deployment..."
        kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
        log_success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|verify|cleanup}"
        echo "Environment variables:"
        echo "  VERSION - Docker image version (default: git commit hash)"
        echo "  ENVIRONMENT - Deployment environment (default: production)"
        echo "  DEPLOYMENT_STRATEGY - rolling|blue-green (default: rolling)"
        echo "  DOCKER_REGISTRY - Docker registry URL (default: localhost:5000)"
        echo "  SKIP_BUILD - Skip Docker build (default: false)"
        echo "  SKIP_PUSH - Skip Docker push (default: false)"
        exit 1
        ;;
esac