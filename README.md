# Fresh Supply Chain Intelligence System
## AI-Powered Food Waste Prediction & Supply Chain Optimization

An enterprise-grade machine learning platform designed to reduce food waste and optimize fresh produce supply chains through computer vision, demand forecasting, and route optimization.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](./.github/workflows/ci-cd.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [ML Models](#ml-models)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)

## Project Overview

The fresh produce industry loses 30% of food through supply chain inefficiencies. This system addresses three critical challenges:

1. **Inconsistent Quality Assessment** - Manual inspection is subjective and time-consuming
2. **Poor Demand Forecasting** - Traditional methods lead to overstock and stockouts  
3. **Inefficient Distribution** - Suboptimal routing increases costs and waste

### Solution Approach

- **Computer Vision**: Automated quality assessment using EfficientNet-based CNN
- **Time Series Forecasting**: Temporal Fusion Transformer for demand prediction
- **Graph Neural Networks**: Optimized distribution routing using GNN + optimization solvers
- **Real-Time Monitoring**: IoT sensor integration for cold chain management

## Key Features

### AI/ML Capabilities
- **Quality Assessment Model**: EfficientNet-B4 CNN for 5-class produce quality prediction
- **Demand Forecasting**: Temporal Fusion Transformer with attention mechanisms
- **Route Optimization**: Graph Neural Network combined with optimization algorithms
- **Anomaly Detection**: Real-time detection of temperature violations and quality issues

### Enterprise Features
- **Production-Ready API**: FastAPI with OAuth2 authentication and rate limiting
- **Real-Time Dashboard**: Interactive Dash/Plotly visualization interface
- **Database Integration**: SQL Server with normalized schema design
- **Caching Layer**: Multi-tier caching with Redis for performance
- **Monitoring Stack**: Prometheus + Grafana + Jaeger for observability
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Containerization**: Docker and Kubernetes deployment configurations

### Data Processing
- **Data Validation**: Comprehensive validation with constraint checking
- **Feature Engineering**: Advanced feature generation for ML models
- **Data Lineage**: Tracking data transformations and dependencies
- **Stream Processing**: Real-time event processing capabilities
- **Error Handling**: Robust error recovery and notification systems

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Client Applications                          │
│         (Web Dashboard, Mobile Apps, Third-Party APIs)           │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                    API Gateway Layer                             │
│  FastAPI + Authentication + Rate Limiting + Load Balancing      │
└────────────────────────┬─────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼──────┐ ┌──────▼─────┐ ┌──────▼──────┐
│  ML Models    │ │   Data     │ │  Monitoring │
│  - Vision     │ │   Layer    │ │  - Metrics  │
│  - Forecast   │ │  - SQL DB  │ │  - Logging  │
│  - GNN Opt    │ │  - Redis   │ │  - Tracing  │
└───────────────┘ └────────────┘ └─────────────┘
```

## Technology Stack

### Core ML/AI
- **Deep Learning**: PyTorch, TensorFlow, TorchVision
- **Computer Vision**: OpenCV, Albumentations, timm
- **Graph ML**: PyTorch Geometric
- **Optimization**: NetworkX, (Gurobi support for advanced optimization)

### Backend & API
- **Framework**: FastAPI 0.105.0
- **Server**: Uvicorn with async support
- **Authentication**: OAuth2 + JWT with passlib
- **Validation**: Pydantic v2.5.0

### Data & Database
- **Database**: SQL Server (pyodbc, pymssql)
- **ORM**: SQLAlchemy 2.0.23
- **Caching**: Redis 5.0.1, aioredis
- **Data Processing**: pandas 2.1.4, numpy 1.24.3

### Monitoring & Observability
- **Metrics**: Prometheus + Grafana
- **Tracing**: OpenTelemetry + Jaeger
- **Logging**: structlog
- **Dashboards**: Plotly + Dash

### DevOps & Deployment
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Testing**: pytest, pytest-asyncio, pytest-cov

## Installation

### Prerequisites
- Python 3.9 or higher
- SQL Server (or compatible database)
- Redis (for caching)
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/saidulIslam1602/Fresh-Supply-Chain-Intelligence---Food-Waste-Prediction-Optimization.git
cd Fresh-Supply-Chain-Intelligence---Food-Waste-Prediction-Optimization
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# For production
pip install -r requirements.txt

# For CI/CD (excludes commercial dependencies)
pip install -r requirements-ci.txt
```

### Step 4: Configure Environment
```bash
cp env.example .env
# Edit .env with your database credentials and API keys
```

### Step 5: Initialize Database
```bash
python scripts/init_database_real.py
python scripts/create_production_schema.py
```

### Step 6: Load Data (Optional)
```bash
# Load real USDA FoodData Central dataset
python load_real_data.py
```

## Usage

### Start Services

#### Option 1: Using Service Manager Script
```bash
python scripts/start_services.py
```

#### Option 2: Manual Start

**Start API Server:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Start Dashboard:**
```bash
python dashboard/app.py
```

### Access Points
- **API Documentation**: http://localhost:8000/docs
- **API Redoc**: http://localhost:8000/redoc  
- **Dashboard**: http://localhost:8050
- **Health Check**: http://localhost:8000/health

### Running Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=api --cov=models --cov=data --cov-report=html
```

## Project Structure

```
.
├── api/                      # FastAPI REST API
│   ├── main.py              # API endpoints and application
│   ├── security.py          # Authentication and authorization
│   └── middleware.py        # Custom middleware
│
├── models/                   # Machine Learning models
│   ├── vision_model.py      # Computer vision for quality assessment
│   ├── forecasting_model.py # Temporal Fusion Transformer
│   ├── gnn_optimizer.py     # Graph Neural Network optimizer
│   └── simple_optimizer.py  # Basic optimization algorithms
│
├── data/                     # Data processing pipeline
│   ├── data_loader.py       # Data loading utilities
│   ├── data_validator.py    # Data validation and quality checks
│   ├── feature_engineer.py  # Feature engineering
│   ├── advanced_preprocessor.py # Advanced preprocessing
│   ├── stream_processor.py  # Real-time stream processing
│   ├── data_lineage.py      # Data lineage tracking
│   └── error_handler.py     # Error handling and recovery
│
├── dashboard/                # Interactive web dashboard
│   ├── app.py               # Dash application
│   └── components/          # Dashboard components
│
├── config/                   # Configuration files
│   └── database_config.py   # Database connection settings
│
├── security/                 # Security modules
│   ├── advanced_auth.py     # Advanced authentication
│   ├── audit_logger.py      # Audit logging
│   └── threat_detector.py   # Threat detection
│
├── performance/              # Performance optimization
│   ├── cache_manager.py     # Multi-tier caching
│   ├── async_processor.py   # Async processing
│   ├── load_balancer.py     # Load balancing
│   └── database_optimizer.py # Database optimization
│
├── monitoring/               # Monitoring and observability
│   ├── prometheus.yml       # Prometheus configuration
│   ├── alertmanager.yml     # Alert manager config
│   ├── grafana/             # Grafana dashboards
│   └── observability/       # Tracing and metrics
│
├── deployment/               # Deployment configurations
│   ├── docker/              # Dockerfiles
│   ├── kubernetes/          # Kubernetes manifests
│   └── scripts/             # Deployment scripts
│
├── tests/                    # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── e2e/                 # End-to-end tests
│   └── performance/         # Performance tests
│
├── scripts/                  # Utility scripts
│   ├── init_database_real.py       # Database initialization
│   ├── load_production_data.py     # Production data loading
│   └── start_services.py           # Service startup manager
│
├── requirements.txt          # Production dependencies
├── requirements-ci.txt       # CI/CD dependencies
└── README.md                 # Project documentation
```

## ML Models

### 1. Computer Vision Quality Assessment

**Architecture**: EfficientNet-B4 based CNN with custom classification head

**Purpose**: Automated produce quality grading

**Classes**:
- Fresh (Grade A)
- Good (Grade B)
- Fair (Grade C)
- Poor (Grade D)
- Expired (Grade F)

**Performance Targets**:
- Inference time: <50ms per image
- Model size: ~50MB
- Input: 224x224 RGB images

### 2. Demand Forecasting

**Architecture**: Temporal Fusion Transformer (TFT)

**Features**:
- Multi-horizon forecasting (1-30 days)
- Attention mechanisms for interpretability
- Uncertainty quantification
- Multiple input modalities (static, temporal, known future)

**Inputs**:
- Historical sales data
- Seasonal patterns
- Weather data
- Promotional calendars
- IoT sensor readings

### 3. Route Optimization

**Architecture**: Graph Neural Network + Optimization Solver

**Components**:
- GCN layers for node embeddings
- Edge importance prediction
- Integration with optimization algorithms

**Optimization Objectives**:
- Minimize transportation costs
- Reduce delivery time
- Maintain cold chain integrity
- Optimize vehicle capacity

## API Endpoints

### Quality Assessment
```
POST /api/v2/predict/quality
- Upload product image
- Returns quality grade and confidence scores
```

### Demand Forecasting
```
POST /api/v2/forecast/demand
- Submit product and timeframe
- Returns demand predictions with uncertainty bounds
```

### Route Optimization
```
POST /api/v2/optimize/routes
- Submit warehouse locations and demand
- Returns optimized delivery routes
```

### Monitoring
```
GET /health              - Health check endpoint
GET /metrics             - Prometheus metrics
GET /api/v2/inventory    - Current inventory status
GET /api/v2/alerts       - Active system alerts
```

## Deployment

### Docker Deployment

**Build Images:**
```bash
docker build -f deployment/docker/Dockerfile.api -t supply-chain-api .
docker build -f deployment/docker/Dockerfile.dashboard -t supply-chain-dashboard .
```

**Run with Docker Compose:**
```bash
cd deployment
docker-compose -f docker-compose.production.yml up -d
```

### Kubernetes Deployment

**Deploy to Kubernetes:**
```bash
# Create namespace and secrets
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/secrets.yaml

# Deploy services
kubectl apply -f deployment/kubernetes/database-deployment.yaml
kubectl apply -f deployment/kubernetes/api-deployment.yaml
kubectl apply -f deployment/kubernetes/dashboard-deployment.yaml
kubectl apply -f deployment/kubernetes/monitoring-deployment.yaml

# Configure ingress
kubectl apply -f deployment/kubernetes/ingress.yaml
```

**Auto-scaling Configuration:**
```bash
# API auto-scales between 3-10 replicas based on CPU utilization
kubectl apply -f deployment/kubernetes/api-deployment.yaml
```

### Monitoring Setup

**Prometheus + Grafana:**
```bash
# Start monitoring stack
kubectl apply -f deployment/kubernetes/monitoring-deployment.yaml

# Access Grafana dashboard
# Default: http://localhost:3000
# Import dashboards from: monitoring/grafana/dashboards/
```

## Configuration

### Environment Variables

Required environment variables (see `.env.example`):

```bash
# Database
SQL_SERVER=localhost
SQL_DATABASE=FreshSupplyChain
SQL_USERNAME=your_username
SQL_PASSWORD=your_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000
JWT_SECRET_KEY=your-secret-key

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Optional: Cloud Storage
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
S3_BUCKET=your-bucket
```

## Performance Optimization

The system includes several performance optimization features:

- **Multi-tier Caching**: In-memory + Redis caching
- **Async Processing**: Non-blocking I/O operations
- **Database Connection Pooling**: Efficient database connections
- **Load Balancing**: Distributed request handling
- **Query Optimization**: Indexed queries and batch operations

## Security Features

- **Authentication**: OAuth2 with JWT tokens
- **Authorization**: Role-based access control (RBAC)
- **Audit Logging**: Comprehensive activity logging
- **Threat Detection**: Real-time security monitoring
- **Encryption**: Data encryption at rest and in transit
- **Rate Limiting**: API request throttling

## Data Sources

### USDA FoodData Central
- **Source**: https://fdc.nal.usda.gov/
- **Dataset**: FoodData Central CSV 2024-04-18
- **Products**: 787,000+ food items with nutritional data
- **Usage**: Product catalog and nutritional information

### Custom Generated Data
- **Warehouse Locations**: Nordic region warehouses (Oslo, Bergen, Trondheim, etc.)
- **IoT Sensor Data**: Temperature, humidity readings
- **Supply Chain Events**: Realistic transaction data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-ci.txt

# Run code quality checks
flake8 .
black --check .
isort --check .
mypy api/ models/ data/

# Run tests with coverage
pytest --cov=. --cov-report=html
```

## Testing

The project includes comprehensive test coverage:

- **Unit Tests**: Individual component testing
- **Integration Tests**: API and database integration
- **E2E Tests**: Complete workflow testing  
- **Performance Tests**: Load and stress testing

Run specific test suites:
```bash
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/e2e/               # End-to-end tests
pytest tests/performance/       # Performance tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- USDA FoodData Central for providing comprehensive food data
- Open-source ML frameworks: PyTorch, TensorFlow, scikit-learn
- FastAPI and Dash communities for excellent frameworks
- Contributors and maintainers of all dependencies

## Contact

Saidul Islam - [@saidulIslam1602](https://github.com/saidulIslam1602)

Project Link: [https://github.com/saidulIslam1602/Fresh-Supply-Chain-Intelligence---Food-Waste-Prediction-Optimization](https://github.com/saidulIslam1602/Fresh-Supply-Chain-Intelligence---Food-Waste-Prediction-Optimization)

---

**Note**: This is a demonstration project showcasing enterprise-grade ML system design and implementation patterns. Performance metrics and business impact calculations are based on industry benchmarks and model capabilities rather than production deployment results.
