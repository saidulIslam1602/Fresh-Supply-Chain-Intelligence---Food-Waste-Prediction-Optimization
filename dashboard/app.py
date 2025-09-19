"""
Enhanced Interactive Dashboard for Fresh Supply Chain Intelligence System
Features: Real-time analytics, AI insights, advanced visualizations, and responsive design
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import asyncio
import websocket
import threading
import time
from sqlalchemy import create_engine, text
import os
import sys
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import redis
from functools import lru_cache

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class DashboardConfig:
    API_BASE_URL: str = "http://localhost:8000"
    WS_URL: str = "ws://localhost:8000/ws/realtime"
    DATABASE_URL: str = "mssql+pyodbc://sa:Saidul1602@localhost:1433/FreshSupplyChain?driver=ODBC+Driver+17+for+SQL+Server"
    REDIS_URL: str = "redis://localhost:6379/0"
    UPDATE_INTERVAL: int = 5000  # 5 seconds
    CACHE_TTL: int = 300  # 5 minutes

config = DashboardConfig()

# Enhanced Dash app with modern theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

app.title = "Fresh Supply Chain Intelligence Dashboard"

# Database and Redis connections
try:
    engine = create_engine(config.DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Successfully connected to SQL Server")
    DB_STATUS = "Connected"
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    engine = None
    DB_STATUS = "Disconnected"

try:
    redis_client = redis.from_url(config.REDIS_URL)
    redis_client.ping()
    logger.info("Successfully connected to Redis")
    CACHE_STATUS = "Connected"
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None
    CACHE_STATUS = "Disconnected"

# Real-time data store
real_time_data = {
    "temperature_logs": [],
    "quality_alerts": [],
    "system_metrics": {},
    "predictions": [],
    "last_update": datetime.now()
}

# WebSocket client for real-time updates
class WebSocketClient:
    def __init__(self, url: str):
        self.url = url
        self.ws = None
        self.running = False
        
    def connect(self):
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            self.running = True
            self.ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
    
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            self.update_real_time_data(data)
        except Exception as e:
            logger.error(f"WebSocket message error: {e}")
    
    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")
        self.running = False
    
    def update_real_time_data(self, data):
        global real_time_data
        if data.get("type") == "system_update":
            real_time_data["system_metrics"] = data.get("data", {})
            real_time_data["last_update"] = datetime.now()

# Start WebSocket client in background thread
ws_client = WebSocketClient(config.WS_URL)
ws_thread = threading.Thread(target=ws_client.connect, daemon=True)
ws_thread.start()

# Enhanced styling
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#F18F01',
    'warning': '#C73E1D',
    'info': '#6C5CE7',
    'light': '#F8F9FA',
    'dark': '#2C3E50',
    'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
}

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .kpi-card {
                background: white;
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                border-left: 4px solid #667eea;
                transition: transform 0.3s ease;
            }
            .kpi-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            }
            .chart-container {
                background: white;
                border-radius: 15px;
                padding: 1rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-online { background-color: #28a745; }
            .status-offline { background-color: #dc3545; }
            .status-warning { background-color: #ffc107; }
            
            .metric-trend {
                font-size: 0.8rem;
                margin-top: 0.5rem;
            }
            .trend-up { color: #28a745; }
            .trend-down { color: #dc3545; }
            .trend-stable { color: #6c757d; }
            
            .alert-badge {
                position: absolute;
                top: -5px;
                right: -5px;
                background: #dc3545;
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                font-size: 0.7rem;
                display: flex;
                align-items: center;
                justify-content: center;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Enhanced layout with modern design
def create_header():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fas fa-leaf me-3"),
                    "Fresh Supply Chain Intelligence"
                ], className="mb-2"),
                html.H4([
                    "AI-Powered Supply Chain Optimization & Food Waste Reduction"
                ], className="mb-3 opacity-75"),
                dbc.Row([
                    dbc.Col([
                        html.Span([
                            html.Span(className=f"status-indicator status-{'online' if DB_STATUS == 'Connected' else 'offline'}"),
                            f"Database: {DB_STATUS}"
                        ], className="me-4"),
                        html.Span([
                            html.Span(className=f"status-indicator status-{'online' if CACHE_STATUS == 'Connected' else 'offline'}"),
                            f"Cache: {CACHE_STATUS}"
                        ], className="me-4"),
                        html.Span([
                            html.Span(className=f"status-indicator status-{'online' if ws_client.running else 'offline'}"),
                            f"Real-time: {'Connected' if ws_client.running else 'Disconnected'}"
                        ])
                    ], width=8),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-clock me-2"),
                            html.Span(id="last-update-time")
                        ], className="text-end")
                    ], width=4)
                ])
            ])
        ])
    ], className="main-header")

def create_kpi_cards():
    return dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-chart-line fa-2x text-primary mb-2"),
                            html.H3(id="kpi-otif", className="text-primary mb-1"),
                            html.P("OTIF Performance", className="mb-1 text-muted"),
                            html.Div(id="otif-trend", className="metric-trend")
                        ])
                    ])
                ], className="kpi-card h-100")
            ])
        ], width=3),
        
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-thermometer-half fa-2x text-info mb-2"),
                            html.H3(id="kpi-temp-compliance", className="text-info mb-1"),
                            html.P("Temperature Compliance", className="mb-1 text-muted"),
                            html.Div(id="temp-trend", className="metric-trend")
                        ])
                    ])
                ], className="kpi-card h-100")
            ])
        ], width=3),
        
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-trash-alt fa-2x text-warning mb-2"),
                            html.H3(id="kpi-waste-reduction", className="text-warning mb-1"),
                            html.P("Waste Reduction", className="mb-1 text-muted"),
                            html.Div(id="waste-trend", className="metric-trend")
                        ])
                    ])
                ], className="kpi-card h-100")
            ])
        ], width=3),
        
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-brain fa-2x text-success mb-2"),
                            html.H3(id="kpi-ai-accuracy", className="text-success mb-1"),
                            html.P("AI Model Accuracy", className="mb-1 text-muted"),
                            html.Div(id="ai-trend", className="metric-trend")
                        ])
                    ])
                ], className="kpi-card h-100")
            ])
        ], width=3)
    ], className="mb-4")

def create_control_panel():
    return dbc.Card([
        dbc.CardHeader([
            html.H5([
                html.I(className="fas fa-cogs me-2"),
                "Control Panel"
            ], className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Time Range", className="fw-bold"),
                    dcc.Dropdown(
                        id="time-range-selector",
                        options=[
                            {"label": "Last Hour", "value": "1h"},
                            {"label": "Last 6 Hours", "value": "6h"},
                            {"label": "Last 24 Hours", "value": "24h"},
                            {"label": "Last 7 Days", "value": "7d"},
                            {"label": "Last 30 Days", "value": "30d"}
                        ],
                        value="24h",
                        clearable=False
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Warehouse", className="fw-bold"),
                    dcc.Dropdown(
                        id="warehouse-selector",
                        options=[
                            {"label": "All Warehouses", "value": "all"},
                            {"label": "Oslo Central", "value": "1"},
                            {"label": "Bergen Hub", "value": "2"},
                            {"label": "Trondheim North", "value": "3"},
                            {"label": "Stavanger South", "value": "4"},
                            {"label": "Tromsø Arctic", "value": "5"}
                        ],
                        value="all",
                        clearable=False
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Product Category", className="fw-bold"),
                    dcc.Dropdown(
                        id="category-selector",
                        options=[
                            {"label": "All Categories", "value": "all"},
                            {"label": "Fruits", "value": "fruits"},
                            {"label": "Vegetables", "value": "vegetables"},
                            {"label": "Dairy", "value": "dairy"},
                            {"label": "Meat & Seafood", "value": "meat"}
                        ],
                        value="all",
                        clearable=False
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Auto Refresh", className="fw-bold"),
                    daq.BooleanSwitch(
                        id="auto-refresh-switch",
                        on=True,
                        color="#28a745",
                        className="mt-2"
                    )
                ], width=3)
            ])
        ])
    ], className="mb-4")

# Main dashboard layout
app.layout = dbc.Container([
    # Header
    create_header(),
    
    # Control Panel
    create_control_panel(),
    
    # KPI Cards
    create_kpi_cards(),
    
    # Main Charts Row 1
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-thermometer-half me-2"),
                            "Real-time Temperature Monitoring"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="temperature-chart", style={"height": "400px"})
                    ])
                ], className="chart-container")
            ])
        ], width=6),
        
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-chart-area me-2"),
                            "AI Demand Forecasting"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="demand-forecast-chart", style={"height": "400px"})
                    ])
                ], className="chart-container")
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Main Charts Row 2
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-eye me-2"),
                            "Computer Vision Quality Assessment"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="quality-assessment-chart", style={"height": "400px"})
                    ])
                ], className="chart-container")
            ])
        ], width=8),
        
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            "Live Alerts"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id="live-alerts", style={"height": "400px", "overflow-y": "auto"})
                    ])
                ], className="chart-container")
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Advanced Analytics Row
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-network-wired me-2"),
                            "Supply Chain Network Optimization"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="network-optimization-chart", style={"height": "500px"})
                    ])
                ], className="chart-container")
            ])
        ], width=6),
        
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-chart-pie me-2"),
                            "Sustainability Impact Dashboard"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="sustainability-chart", style={"height": "500px"})
                    ])
                ], className="chart-container")
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Data Tables Row
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-table me-2"),
                            "Recent Predictions & Analytics"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dash_table.DataTable(
                            id="predictions-table",
                            columns=[
                                {"name": "Timestamp", "id": "timestamp"},
                                {"name": "Product", "id": "product"},
                                {"name": "Quality Score", "id": "quality_score"},
                                {"name": "Predicted Demand", "id": "predicted_demand"},
                                {"name": "Confidence", "id": "confidence"},
                                {"name": "Action Required", "id": "action"}
                            ],
                            style_cell={'textAlign': 'left', 'padding': '10px'},
                            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{quality_score} < 0.5'},
                                    'backgroundColor': '#f8d7da',
                                    'color': 'black',
                                },
                                {
                                    'if': {'filter_query': '{quality_score} >= 0.8'},
                                    'backgroundColor': '#d4edda',
                                    'color': 'black',
                                }
                            ],
                            page_size=10,
                            sort_action="native"
                        )
                    ])
                ], className="chart-container")
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Intervals for real-time updates
    dcc.Interval(id="main-interval", interval=config.UPDATE_INTERVAL, n_intervals=0),
    dcc.Interval(id="kpi-interval", interval=10000, n_intervals=0),  # KPIs every 10 seconds
    dcc.Interval(id="alerts-interval", interval=3000, n_intervals=0),  # Alerts every 3 seconds
    
    # Store components for data sharing
    dcc.Store(id="dashboard-data-store"),
    dcc.Store(id="user-preferences-store")
    
], fluid=True, className="px-4")

# Enhanced data fetching functions
@lru_cache(maxsize=128)
def fetch_api_data(endpoint: str, params: str = "") -> Dict[str, Any]:
    """Fetch data from enhanced API with caching"""
    try:
        url = f"{config.API_BASE_URL}{endpoint}"
        if params:
            url += f"?{params}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API request failed for {endpoint}: {e}")
        return {}

def get_cached_data(key: str, fetch_func, ttl: int = config.CACHE_TTL):
    """Get data with Redis caching"""
    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache read error: {e}")
    
    # Fetch fresh data
    data = fetch_func()
    
    # Cache the result
    if redis_client and data:
        try:
            redis_client.setex(key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Cache write error: {e}")
    
    return data

def generate_mock_data():
    """Generate realistic mock data for demonstration"""
    current_time = datetime.now()
    
    # Mock temperature data
    temp_data = []
    for i in range(100):
        timestamp = current_time - timedelta(minutes=i*5)
        temp_data.append({
            "timestamp": timestamp.isoformat(),
            "warehouse_id": np.random.choice([1, 2, 3, 4, 5]),
            "temperature": np.random.normal(4, 2),  # Target: 2-6°C
            "humidity": np.random.normal(85, 10),
            "quality_score": max(0, min(1, np.random.normal(0.8, 0.15)))
        })
    
    # Mock predictions
    predictions_data = []
    products = ["Organic Apples", "Fresh Salmon", "Leafy Greens", "Dairy Milk", "Berries Mix"]
    for i in range(20):
        predictions_data.append({
            "timestamp": (current_time - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M"),
            "product": np.random.choice(products),
            "quality_score": round(np.random.uniform(0.3, 0.95), 2),
            "predicted_demand": round(np.random.uniform(50, 500), 0),
            "confidence": round(np.random.uniform(0.7, 0.98), 2),
            "action": np.random.choice(["Monitor", "Urgent Review", "Optimize Route", "Quality Check"])
        })
    
    return {
        "temperature_data": temp_data,
        "predictions_data": predictions_data,
        "kpis": {
            "otif_rate": round(np.random.uniform(92, 98), 1),
            "temp_compliance": round(np.random.uniform(88, 96), 1),
            "waste_reduction": round(np.random.uniform(15, 35), 1),
            "ai_accuracy": round(np.random.uniform(89, 96), 1)
        }
    }

# Callback for updating last update time
@app.callback(
    Output("last-update-time", "children"),
    Input("main-interval", "n_intervals")
)
def update_last_update_time(n):
    return datetime.now().strftime("%H:%M:%S")

# Enhanced KPI callback with trends
@app.callback(
    [Output("kpi-otif", "children"),
     Output("kpi-temp-compliance", "children"), 
     Output("kpi-waste-reduction", "children"),
     Output("kpi-ai-accuracy", "children"),
     Output("otif-trend", "children"),
     Output("temp-trend", "children"),
     Output("waste-trend", "children"),
     Output("ai-trend", "children")],
    Input("kpi-interval", "n_intervals")
)
def update_enhanced_kpis(n):
    """Update KPIs with trend indicators"""
    
    def get_kpi_data():
        if engine:
            try:
                # Real KPI calculations would go here
                pass
            except Exception as e:
                logger.error(f"KPI calculation error: {e}")
        
        # Use mock data for demonstration
        mock_data = generate_mock_data()
        return mock_data["kpis"]
    
    kpis = get_cached_data("dashboard:kpis", get_kpi_data, 60)
    
    # Generate trend indicators
    def create_trend_indicator(value, target, is_higher_better=True):
        if is_higher_better:
            if value >= target:
                return html.Span([
                    html.I(className="fas fa-arrow-up me-1"),
                    f"+{value-target:.1f}% vs target"
                ], className="trend-up")
            else:
                return html.Span([
                    html.I(className="fas fa-arrow-down me-1"),
                    f"{value-target:.1f}% vs target"
                ], className="trend-down")
        else:
            if value <= target:
                return html.Span([
                    html.I(className="fas fa-arrow-down me-1"),
                    f"{target-value:.1f}% improvement"
                ], className="trend-up")
            else:
                return html.Span([
                    html.I(className="fas fa-arrow-up me-1"),
                    f"+{value-target:.1f}% vs target"
                ], className="trend-down")
    
    return (
        f"{kpis['otif_rate']}%",
        f"{kpis['temp_compliance']}%", 
        f"{kpis['waste_reduction']}%",
        f"{kpis['ai_accuracy']}%",
        create_trend_indicator(kpis['otif_rate'], 95),
        create_trend_indicator(kpis['temp_compliance'], 90),
        create_trend_indicator(kpis['waste_reduction'], 25, False),
        create_trend_indicator(kpis['ai_accuracy'], 90)
    )

# Enhanced temperature monitoring chart
@app.callback(
    Output("temperature-chart", "figure"),
    [Input("main-interval", "n_intervals"),
     Input("time-range-selector", "value"),
     Input("warehouse-selector", "value")]
)
def update_temperature_chart(n, time_range, warehouse):
    """Create enhanced temperature monitoring chart"""
    
    def get_temperature_data():
        mock_data = generate_mock_data()
        return mock_data["temperature_data"]
    
    temp_data = get_cached_data(f"dashboard:temperature:{time_range}:{warehouse}", get_temperature_data, 30)
    
    if not temp_data:
        return go.Figure()
    
    df = pd.DataFrame(temp_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter by warehouse if specified
    if warehouse != "all":
        df = df[df['warehouse_id'] == int(warehouse)]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Temperature & Humidity Monitoring", "Quality Score Trend"),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Temperature trace
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['temperature'],
            mode='lines+markers',
            name='Temperature (°C)',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Temperature</b><br>%{y:.1f}°C<br>%{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add temperature threshold zones
    fig.add_hrect(
        y0=2, y1=6,
        fillcolor="rgba(40, 167, 69, 0.2)",
        layer="below",
        line_width=0,
        annotation_text="Optimal Range",
        annotation_position="top left",
        row=1, col=1
    )
    
    # Humidity trace (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['humidity'],
            mode='lines',
            name='Humidity (%)',
            line=dict(color='#4ECDC4', width=2, dash='dash'),
            yaxis='y2',
            hovertemplate='<b>Humidity</b><br>%{y:.1f}%<br>%{x}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Quality score
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['quality_score'],
            mode='lines+markers',
            name='Quality Score',
            line=dict(color='#A8E6CF', width=3),
            marker=dict(size=6, symbol='diamond'),
            fill='tonexty',
            hovertemplate='<b>Quality Score</b><br>%{y:.2f}<br>%{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Humidity (%)", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Quality Score", row=2, col=1)
    
    return fig

# Enhanced demand forecasting chart
@app.callback(
    Output("demand-forecast-chart", "figure"),
    [Input("main-interval", "n_intervals"),
     Input("category-selector", "value")]
)
def update_demand_forecast_chart(n, category):
    """Create enhanced demand forecasting chart with uncertainty bands"""
    
    # Generate mock forecast data
    dates = pd.date_range(start=datetime.now(), periods=14, freq='D')
    base_demand = 100
    
    # Historical data
    hist_dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=7, freq='D')
    historical_demand = [base_demand + np.random.normal(0, 10) + 5 * np.sin(i * 0.5) for i in range(7)]
    
    # Forecast data with uncertainty
    forecast_demand = [base_demand + np.random.normal(0, 5) + 10 * np.sin(i * 0.3) for i in range(14)]
    upper_bound = [d * 1.2 for d in forecast_demand]
    lower_bound = [d * 0.8 for d in forecast_demand]
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=historical_demand,
        mode='lines+markers',
        name='Historical Demand',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8)
    ))
    
    # Forecast uncertainty band
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=dates,
        y=forecast_demand,
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color='#F18F01', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Add vertical line for current date
    fig.add_vline(
        x=datetime.now(),
        line_dash="solid",
        line_color="red",
        annotation_text="Today",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="7-Day Demand Forecast with Uncertainty Quantification",
        xaxis_title="Date",
        yaxis_title="Demand (Units)",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

# Quality assessment chart
@app.callback(
    Output("quality-assessment-chart", "figure"),
    Input("main-interval", "n_intervals")
)
def update_quality_assessment_chart(n):
    """Create computer vision quality assessment visualization"""
    
    # Mock quality distribution data
    quality_labels = ['Fresh', 'Good', 'Fair', 'Poor', 'Spoiled']
    quality_counts = [45, 30, 15, 8, 2]
    quality_colors = ['#28a745', '#17a2b8', '#ffc107', '#fd7e14', '#dc3545']
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=quality_labels,
        values=quality_counts,
        hole=0.4,
        marker_colors=quality_colors,
        textinfo='label+percent',
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    # Add center text
    fig.add_annotation(
        text=f"<b>Total<br>Inspections</b><br>{sum(quality_counts)}",
        x=0.5, y=0.5,
        font_size=16,
        showarrow=False
    )
    
    fig.update_layout(
        title="Real-time Quality Assessment Distribution",
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Live alerts
@app.callback(
    Output("live-alerts", "children"),
    Input("alerts-interval", "n_intervals")
)
def update_live_alerts(n):
    """Update live alerts panel"""
    
    # Mock alerts data
    alerts = [
        {
            "time": "2 min ago",
            "type": "warning",
            "message": "Temperature spike detected in Warehouse 2",
            "icon": "fas fa-thermometer-full"
        },
        {
            "time": "5 min ago", 
            "type": "success",
            "message": "Quality inspection completed - 98% accuracy",
            "icon": "fas fa-check-circle"
        },
        {
            "time": "8 min ago",
            "type": "info",
            "message": "New demand forecast generated for next week",
            "icon": "fas fa-chart-line"
        },
        {
            "time": "12 min ago",
            "type": "danger",
            "message": "Critical: Spoilage risk detected in Batch #A2024",
            "icon": "fas fa-exclamation-triangle"
        },
        {
            "time": "15 min ago",
            "type": "success",
            "message": "Route optimization completed - 15% cost reduction",
            "icon": "fas fa-route"
        }
    ]
    
    alert_components = []
    for alert in alerts:
        color_map = {
            "success": "success",
            "warning": "warning", 
            "info": "info",
            "danger": "danger"
        }
        
        alert_components.append(
            dbc.Alert([
                html.I(className=f"{alert['icon']} me-2"),
                html.Strong(alert['message']),
                html.Small(f" • {alert['time']}", className="text-muted ms-2")
            ], color=color_map[alert['type']], className="mb-2 py-2")
        )
    
    return alert_components

# Network optimization chart
@app.callback(
    Output("network-optimization-chart", "figure"),
    Input("main-interval", "n_intervals")
)
def update_network_optimization_chart(n):
    """Create supply chain network optimization visualization"""
    
    # Mock network data
    warehouses = {
        'Oslo': {'lat': 59.9139, 'lon': 10.7522, 'size': 100, 'efficiency': 0.95},
        'Bergen': {'lat': 60.3913, 'lon': 5.3221, 'size': 80, 'efficiency': 0.88},
        'Trondheim': {'lat': 63.4305, 'lon': 10.3951, 'size': 60, 'efficiency': 0.92},
        'Stavanger': {'lat': 58.9700, 'lon': 5.7331, 'size': 70, 'efficiency': 0.90},
        'Tromsø': {'lat': 69.6492, 'lon': 18.9553, 'size': 40, 'efficiency': 0.85}
    }
    
    # Create map
    fig = go.Figure()
    
    # Add warehouse locations
    for name, data in warehouses.items():
        fig.add_trace(go.Scattermapbox(
            lat=[data['lat']],
            lon=[data['lon']],
            mode='markers',
            marker=dict(
                size=data['size']/5,
                color=data['efficiency'],
                colorscale='RdYlGn',
                cmin=0.8,
                cmax=1.0,
                showscale=True,
                colorbar=dict(title="Efficiency Score")
            ),
            text=f"{name}<br>Efficiency: {data['efficiency']:.1%}",
            hovertemplate='<b>%{text}</b><extra></extra>',
            name=name
        ))
    
    # Add route connections
    connections = [
        ('Oslo', 'Bergen'), ('Oslo', 'Trondheim'), 
        ('Bergen', 'Stavanger'), ('Trondheim', 'Tromsø')
    ]
    
    for start, end in connections:
        fig.add_trace(go.Scattermapbox(
            lat=[warehouses[start]['lat'], warehouses[end]['lat']],
            lon=[warehouses[start]['lon'], warehouses[end]['lon']],
            mode='lines',
            line=dict(width=3, color='rgba(46, 134, 171, 0.6)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=63.0, lon=10.0),
            zoom=4
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig

# Sustainability chart
@app.callback(
    Output("sustainability-chart", "figure"),
    Input("main-interval", "n_intervals")
)
def update_sustainability_chart(n):
    """Create sustainability impact visualization"""
    
    # Mock sustainability metrics
    metrics = ['CO2 Reduction', 'Waste Reduction', 'Energy Savings', 'Water Conservation']
    current = [25, 30, 18, 22]  # Percentage improvements
    target = [30, 35, 25, 28]
    
    fig = go.Figure()
    
    # Current performance
    fig.add_trace(go.Bar(
        x=metrics,
        y=current,
        name='Current Performance',
        marker_color='#28a745',
        text=[f'{v}%' for v in current],
        textposition='auto'
    ))
    
    # Target performance
    fig.add_trace(go.Bar(
        x=metrics,
        y=target,
        name='Target',
        marker_color='rgba(40, 167, 69, 0.3)',
        text=[f'{v}%' for v in target],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Sustainability Impact Metrics",
        xaxis_title="Sustainability Metrics",
        yaxis_title="Improvement (%)",
        height=500,
        barmode='group',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

# Predictions table
@app.callback(
    Output("predictions-table", "data"),
    Input("main-interval", "n_intervals")
)
def update_predictions_table(n):
    """Update predictions and analytics table"""
    
    mock_data = generate_mock_data()
    return mock_data["predictions_data"]

if __name__ == "__main__":
    app.run_server(
        debug=True,
        host="0.0.0.0",
        port=3000,
        dev_tools_hot_reload=True,
        dev_tools_ui=True
    )