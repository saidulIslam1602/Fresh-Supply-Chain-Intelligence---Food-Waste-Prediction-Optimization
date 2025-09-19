# Fresh Supply Chain Intelligence System

AI-powered solution for food waste reduction and supply chain optimization in the fresh produce industry.

## Business Problem

The fresh produce industry faces significant challenges that impact profitability and sustainability:
- 30% food waste across the supply chain
- Manual quality assessment leading to inconsistent decisions
- Poor demand forecasting causing stockouts and overstock
- Temperature violations compromising cold chain integrity
- Suboptimal routing increasing transportation costs

## AI Solution

Four core capabilities that directly address these problems:

| Problem | AI Solution | Implementation |
|---------|-------------|----------------|
| Food Waste | Computer Vision Quality Assessment | EfficientNet-B4 with 5-class quality prediction |
| Poor Forecasting | Time Series AI (TFT) | Temporal Fusion Transformer with uncertainty quantification |
| Inefficient Routes | Graph Neural Network Optimization | GNN + Gurobi solver for route optimization |
| Temperature Issues | Real-time IoT Monitoring | WebSocket-based live monitoring system |

## System Architecture

```
AI/ML Core              Real Data Foundation        Production API
├── Computer Vision     ├── 787,526 USDA Products  ├── Quality Assessment
├── Time Series AI      ├── 5 Warehouse Locations   ├── Demand Forecasting  
├── Route Optimization  ├── IoT Sensor Network      ├── Route Optimization
└── Real-time Analytics └── Supply Chain Events    └── Live Monitoring
```

## Quick Start

```bash
# Clone and setup
git clone <repository>
cd Fresh-Supply-Chain-Intelligence
cp env.example .env

# Start all services
docker-compose up -d

# Load real data
python load_real_data.py

# Access services
# API: http://localhost:8000/docs
# Dashboard: http://localhost:3000
```

## Core AI Solutions

### 1. Quality Assessment API
```bash
POST /api/v1/predict/quality
# Input: Product image, lot number, warehouse
# Output: Quality score (0-1), freshness label (Fresh/Good/Fair/Poor/Spoiled)
# Model: EfficientNet-B4 with custom classifier
```

### 2. Demand Forecasting API
```bash
POST /api/v1/forecast/demand  
# Input: Product ID, warehouse, forecast horizon
# Output: 7-day demand forecast with confidence intervals
# Model: Temporal Fusion Transformer with attention mechanism
```

### 3. Route Optimization API
```bash
POST /api/v1/optimize/distribution
# Input: Products, warehouses, constraints
# Output: Optimal routes, cost savings, waste reduction
# Model: Graph Neural Network + Gurobi optimization
```

### 4. Real-time Monitoring
```bash
WebSocket /ws/temperature-monitor
# Output: Live temperature, humidity, quality alerts
# Implementation: Real-time IoT data streaming
```

## Technical Implementation

### API Endpoints (10 total)
- GET / - Root endpoint with system status
- GET /api/v1/products - USDA product catalog
- GET /api/v1/warehouses - Warehouse locations
- GET /api/v1/iot/readings - IoT sensor data
- GET /api/v1/analytics/categories - Product analytics
- POST /api/v1/predict/quality - Quality prediction
- POST /api/v1/forecast/demand - Demand forecasting
- POST /api/v1/optimize/distribution - Route optimization
- GET /api/v1/metrics/kpi - Key performance indicators
- WebSocket /ws/temperature-monitor - Real-time monitoring

### ML Models
- **Vision Model**: EfficientNet-B4 with custom classifier (5 quality classes)
- **Forecasting Model**: Temporal Fusion Transformer (256 hidden units, 8 attention heads)
- **Optimization Model**: Graph Neural Network with Gurobi solver
- **Real-time Analytics**: IoT data processing with quality scoring

### Data Foundation
- **787,526 USDA Products**: Real food database with nutritional information
- **5 Warehouse Locations**: Configurable warehouse network
- **IoT Sensor Network**: Temperature, humidity, CO2, ethylene monitoring
- **Supply Chain Events**: Waste tracking and transportation data

## Technical Stack

Project Structure:
- api/ - FastAPI application with 10 endpoints
- models/ - ML models (Vision, TFT, GNN, Optimization)
- data/ - Real USDA data loader and processing
- config/ - Database (SQL Server) & Redis configuration
- monitoring/ - Prometheus + Grafana dashboards
- tests/ - Test suite covering API and models

Technology Stack:
- Backend: FastAPI, Python 3.11+, SQL Server, Redis
- ML/AI: PyTorch, TensorFlow, scikit-learn, Gurobi
- Monitoring: Prometheus, Grafana, Docker
- Testing: pytest with API and model tests

## Data Sources

Real Data Integration:
- **USDA FoodData Central**: 787,526 real food products with nutritional data
- **Warehouse Network**: 5 configurable locations with capacity and coordinates
- **IoT Simulation**: Norwegian climate-based sensor data generation
- **Supply Chain Modeling**: Realistic waste patterns and transportation costs

Database Schema:
- Products table with real USDA data
- Warehouses with location and capacity information
- TemperatureLogs for IoT sensor data (time-series optimized)
- WasteEvents for tracking and analysis
- Inventory management with lot tracking

## Deployment

Production Deployment:
```bash
# Docker deployment
docker-compose up -d

# Local development  
pip install -r requirements.txt
python scripts/start_services.py

# Run tests
pytest tests/ -v
```

Key Features:
- Containerized deployment with Docker Compose
- SQL Server database with real USDA data
- Redis caching for performance
- JWT authentication for API security
- Comprehensive test suite for reliability
- Production-ready monitoring with Prometheus/Grafana

## Performance Characteristics

API Performance:
- Response time target: <200ms for predictions
- Concurrent request handling with FastAPI
- Redis caching for frequently accessed data
- Database optimized for time-series queries

Model Performance:
- Computer Vision: 5-class quality classification
- Time Series: 7-day forecast horizon with uncertainty
- Optimization: Multi-constraint route planning
- Real-time: WebSocket streaming for live data

Built for demonstrating advanced data science capabilities in supply chain optimization and food waste reduction.