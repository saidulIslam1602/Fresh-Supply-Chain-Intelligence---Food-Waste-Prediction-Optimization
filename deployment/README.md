# Fresh Supply Chain Intelligence - Deployment Guide

## Overview

This directory contains comprehensive deployment configurations for the Fresh Supply Chain Intelligence System, supporting both Kubernetes and Docker Compose deployments with enterprise-grade features.

## Deployment Options

### 1. Kubernetes Deployment (Recommended for Production)

**Features:**
- Auto-scaling with HPA
- Rolling updates and blue-green deployments
- Health checks and self-healing
- Resource quotas and limits
- Network policies and security
- Persistent storage
- Service mesh ready

**Quick Start:**
```bash
# Deploy to Kubernetes
./deployment/scripts/deploy.sh

# Deploy with blue-green strategy
DEPLOYMENT_STRATEGY=blue-green ./deployment/scripts/deploy.sh

# Rollback deployment
./deployment/scripts/deploy.sh rollback

# Cleanup
./deployment/scripts/deploy.sh cleanup
```

### 2. Docker Compose Deployment

**Features:**
- Multi-container orchestration
- Volume persistence
- Health checks
- Resource limits
- Network isolation
- Production-ready configuration

**Quick Start:**
```bash
# Production deployment
docker-compose -f deployment/docker-compose.production.yml up -d

# View logs
docker-compose -f deployment/docker-compose.production.yml logs -f

# Scale services
docker-compose -f deployment/docker-compose.production.yml up -d --scale api=3 --scale dashboard=2
```

## Architecture

### Kubernetes Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ingress       │    │   Load Balancer │    │   Auto Scaler   │
│   (Nginx)       │    │   (Service)     │    │   (HPA)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
┌───▼────┐  ┌─────────┐  ┌──────▼──┐  ┌─────────┐  ┌─────────┐
│   API  │  │Dashboard│  │ Monitor │  │Database │  │  Cache  │
│ (3-10) │  │  (2-6)  │  │ Stack   │  │(StatefulSet)│(Redis)│
└────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

### Components

#### Application Services
- **API Service**: FastAPI application with auto-scaling (3-10 replicas)
- **Dashboard Service**: Plotly Dash with auto-scaling (2-6 replicas)
- **Database**: SQL Server StatefulSet with persistent storage
- **Cache**: Redis with persistence and monitoring

#### Infrastructure Services
- **Monitoring**: Prometheus, Grafana, Jaeger, Alertmanager
- **Networking**: Nginx Ingress, Network Policies, Service Mesh
- **Storage**: Persistent Volumes with fast SSD storage class
- **Security**: RBAC, Pod Security Policies, Network Policies

## Configuration

### Environment Variables

**Required:**
```bash
DATABASE_URL=mssql+pyodbc://sa:password@sqlserver:1433/FreshSupplyChainDB
JWT_SECRET_KEY=your-super-secret-jwt-key
DB_PASSWORD=YourStrongPassword123!
```

**Optional:**
```bash
ENVIRONMENT=production
DEPLOYMENT_STRATEGY=rolling  # or blue-green
DOCKER_REGISTRY=your-registry.com
VERSION=v2.0.0
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin123
```

### Resource Requirements

#### Minimum Requirements
- **CPU**: 4 cores
- **Memory**: 8GB RAM
- **Storage**: 100GB SSD
- **Network**: 1Gbps

#### Recommended Production
- **CPU**: 16 cores
- **Memory**: 32GB RAM
- **Storage**: 500GB NVMe SSD
- **Network**: 10Gbps

### Scaling Configuration

#### Horizontal Pod Autoscaler (HPA)
```yaml
# API Service
minReplicas: 3
maxReplicas: 10
targetCPUUtilization: 70%
targetMemoryUtilization: 80%

# Dashboard Service
minReplicas: 2
maxReplicas: 6
targetCPUUtilization: 70%
targetMemoryUtilization: 80%
```

#### Vertical Pod Autoscaler (VPA)
```yaml
# Resource Requests/Limits
API:
  requests: { cpu: 200m, memory: 512Mi }
  limits: { cpu: 1000m, memory: 2Gi }

Dashboard:
  requests: { cpu: 100m, memory: 256Mi }
  limits: { cpu: 500m, memory: 1Gi }
```

## Security

### Network Security
- **Network Policies**: Restrict pod-to-pod communication
- **Ingress Security**: TLS termination, rate limiting
- **Service Mesh**: mTLS, traffic policies (optional)

### Pod Security
- **Security Contexts**: Non-root users, read-only filesystems
- **Pod Security Policies**: Restricted capabilities
- **RBAC**: Least privilege access

### Data Security
- **Secrets Management**: Kubernetes secrets, external secret operators
- **Encryption**: At-rest and in-transit encryption
- **Audit Logging**: Complete audit trail

## Monitoring & Observability

### Metrics Collection
- **Prometheus**: System and application metrics
- **Node Exporter**: Host-level metrics
- **cAdvisor**: Container metrics
- **Custom Metrics**: Business KPIs

### Visualization
- **Grafana**: Dashboards and alerting
- **Jaeger**: Distributed tracing
- **Kibana**: Log analysis (optional)

### Alerting
- **Alertmanager**: Alert routing and management
- **Multi-channel**: Email, Slack, PagerDuty
- **SLA Monitoring**: Uptime and performance SLAs

## Deployment Strategies

### Rolling Deployment (Default)
- **Zero Downtime**: Gradual replacement of pods
- **Rollback**: Automatic rollback on failure
- **Health Checks**: Readiness and liveness probes

### Blue-Green Deployment
- **Instant Switch**: Complete environment switch
- **Risk Mitigation**: Full testing before switch
- **Quick Rollback**: Instant rollback capability

### Canary Deployment (Advanced)
- **Gradual Rollout**: Traffic splitting
- **A/B Testing**: Performance comparison
- **Automated Promotion**: Based on metrics

## Backup & Recovery

### Database Backup
```bash
# Automated daily backups
kubectl create cronjob db-backup --image=backup-tool --schedule="0 2 * * *"
```

### Application State
- **Persistent Volumes**: Automatic snapshots
- **Configuration**: GitOps with version control
- **Secrets**: External secret management

### Disaster Recovery
- **Multi-Region**: Cross-region replication
- **RTO**: Recovery Time Objective < 1 hour
- **RPO**: Recovery Point Objective < 15 minutes

## Troubleshooting

### Common Issues

#### Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n fresh-supply-chain

# Check pod logs
kubectl logs -f deployment/api-deployment -n fresh-supply-chain

# Describe pod for events
kubectl describe pod <pod-name> -n fresh-supply-chain
```

#### Service Discovery Issues
```bash
# Check services
kubectl get svc -n fresh-supply-chain

# Check endpoints
kubectl get endpoints -n fresh-supply-chain

# Test service connectivity
kubectl run debug --rm -i --tty --image=nicolaka/netshoot
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n fresh-supply-chain
kubectl top nodes

# Check HPA status
kubectl get hpa -n fresh-supply-chain

# Check metrics
curl http://prometheus:9090/api/v1/query?query=up
```

### Debug Commands

```bash
# Get all resources
kubectl get all -n fresh-supply-chain

# Check resource quotas
kubectl describe quota -n fresh-supply-chain

# Check network policies
kubectl get networkpolicy -n fresh-supply-chain

# Port forward for local access
kubectl port-forward svc/api-service 8000:8000 -n fresh-supply-chain
kubectl port-forward svc/grafana-service 3000:3000 -n fresh-supply-chain
```

## Performance Tuning

### Database Optimization
- **Connection Pooling**: Optimized pool sizes
- **Query Optimization**: Indexed queries
- **Resource Allocation**: Dedicated CPU/Memory

### Cache Optimization
- **Redis Configuration**: Memory policies, persistence
- **Cache Strategies**: Multi-tier caching
- **Eviction Policies**: LRU, TTL-based

### Application Optimization
- **JVM Tuning**: Garbage collection optimization
- **Connection Pools**: Database and HTTP pools
- **Async Processing**: Non-blocking operations

## Maintenance

### Regular Tasks
- **Security Updates**: Monthly OS and dependency updates
- **Certificate Renewal**: Automated with cert-manager
- **Backup Verification**: Weekly backup restoration tests
- **Performance Review**: Monthly performance analysis

### Upgrade Process
1. **Staging Deployment**: Test in staging environment
2. **Blue-Green Switch**: Deploy to production
3. **Health Verification**: Comprehensive health checks
4. **Rollback Plan**: Ready rollback procedure

## Support

### Documentation
- **API Documentation**: OpenAPI/Swagger
- **Runbooks**: Operational procedures
- **Architecture Diagrams**: System design docs

### Monitoring Dashboards
- **System Overview**: http://grafana.fresh-supply-chain.local
- **Application Metrics**: Custom business dashboards
- **Infrastructure Health**: Node and cluster metrics

### Contact Information
- **Development Team**: dev-team@company.com
- **Operations Team**: ops-team@company.com
- **Emergency Contact**: on-call@company.com