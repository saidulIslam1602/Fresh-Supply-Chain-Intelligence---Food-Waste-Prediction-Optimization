"""
Advanced Chart Components for Enhanced Dashboard
Reusable chart components with consistent styling and interactivity
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Color palette for consistent theming
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'gradient_colors': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
}

class ChartFactory:
    """Factory class for creating standardized charts"""
    
    @staticmethod
    def get_base_layout(title: str = "", height: int = 400) -> Dict[str, Any]:
        """Get base layout configuration for all charts"""
        return {
            'title': {
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': COLORS['dark']}
            },
            'height': height,
            'margin': {'l': 50, 'r': 50, 't': 60, 'b': 50},
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': COLORS['dark']},
            'showlegend': True,
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1
            }
        }
    
    @staticmethod
    def style_axes(fig: go.Figure, grid_color: str = 'rgba(128,128,128,0.2)'):
        """Apply consistent axis styling"""
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            linecolor='rgba(128,128,128,0.5)',
            linewidth=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=grid_color,
            linecolor='rgba(128,128,128,0.5)',
            linewidth=1
        )
        return fig

class TemperatureCharts:
    """Temperature monitoring chart components"""
    
    @staticmethod
    def create_real_time_temperature_chart(
        data: List[Dict[str, Any]], 
        warehouse_filter: Optional[str] = None,
        time_range: str = "24h"
    ) -> go.Figure:
        """Create real-time temperature monitoring chart with multiple metrics"""
        
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by warehouse if specified
        if warehouse_filter and warehouse_filter != "all":
            df = df[df['warehouse_id'] == int(warehouse_filter)]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Temperature Monitoring with Alerts",
                "Humidity & Quality Correlation", 
                "Temperature Distribution by Warehouse"
            ),
            vertical_spacing=0.08,
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": False}]
            ]
        )
        
        # Temperature with threshold zones
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['temperature'],
                mode='lines+markers',
                name='Temperature (°C)',
                line=dict(color=COLORS['danger'], width=2),
                marker=dict(size=4),
                hovertemplate='<b>Temperature</b><br>%{y:.1f}°C<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add temperature zones
        fig.add_hrect(
            y0=2, y1=6,
            fillcolor="rgba(40, 167, 69, 0.2)",
            layer="below",
            line_width=0,
            annotation_text="Optimal Zone",
            annotation_position="top left",
            row=1, col=1
        )
        
        fig.add_hrect(
            y0=-2, y1=2,
            fillcolor="rgba(255, 193, 7, 0.2)",
            layer="below",
            line_width=0,
            annotation_text="Warning Zone",
            annotation_position="bottom left",
            row=1, col=1
        )
        
        # Humidity on secondary axis
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['humidity'],
                mode='lines',
                name='Humidity (%)',
                line=dict(color=COLORS['info'], width=2, dash='dash'),
                yaxis='y2',
                hovertemplate='<b>Humidity</b><br>%{y:.1f}%<br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Quality score correlation
        fig.add_trace(
            go.Scatter(
                x=df['humidity'],
                y=df['quality_score'],
                mode='markers',
                name='Quality vs Humidity',
                marker=dict(
                    size=8,
                    color=df['temperature'],
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Temp (°C)", x=1.02)
                ),
                hovertemplate='<b>Quality Score</b><br>%{y:.2f}<br>Humidity: %{x:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Temperature distribution by warehouse
        if 'warehouse_id' in df.columns:
            for warehouse_id in df['warehouse_id'].unique():
                warehouse_data = df[df['warehouse_id'] == warehouse_id]
                fig.add_trace(
                    go.Box(
                        y=warehouse_data['temperature'],
                        name=f'Warehouse {warehouse_id}',
                        boxpoints='outliers',
                        marker_color=COLORS['gradient_colors'][warehouse_id % len(COLORS['gradient_colors'])]
                    ),
                    row=3, col=1
                )
        
        # Update layout
        layout = ChartFactory.get_base_layout("Real-time Temperature Monitoring Dashboard", 800)
        fig.update_layout(**layout)
        
        # Update axes
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Humidity (%)", secondary_y=True, row=1, col=1)
        fig.update_xaxes(title_text="Humidity (%)", row=2, col=1)
        fig.update_yaxes(title_text="Quality Score", row=2, col=1)
        fig.update_xaxes(title_text="Warehouse", row=3, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=3, col=1)
        
        return ChartFactory.style_axes(fig)

class ForecastingCharts:
    """Demand forecasting chart components"""
    
    @staticmethod
    def create_advanced_forecast_chart(
        historical_data: List[Dict[str, Any]],
        forecast_data: List[Dict[str, Any]],
        confidence_intervals: Optional[Dict[str, List[float]]] = None
    ) -> go.Figure:
        """Create advanced demand forecasting chart with uncertainty quantification"""
        
        fig = go.Figure()
        
        if historical_data:
            hist_df = pd.DataFrame(historical_data)
            hist_df['date'] = pd.to_datetime(hist_df['date'])
            
            # Historical demand
            fig.add_trace(go.Scatter(
                x=hist_df['date'],
                y=hist_df['demand'],
                mode='lines+markers',
                name='Historical Demand',
                line=dict(color=COLORS['primary'], width=3),
                marker=dict(size=6),
                hovertemplate='<b>Historical</b><br>%{y:.0f} units<br>%{x}<extra></extra>'
            ))
        
        if forecast_data:
            forecast_df = pd.DataFrame(forecast_data)
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            # Confidence intervals
            if confidence_intervals:
                # Upper and lower bounds
                fig.add_trace(go.Scatter(
                    x=list(forecast_df['date']) + list(forecast_df['date'][::-1]),
                    y=confidence_intervals['upper'] + confidence_intervals['lower'][::-1],
                    fill='toself',
                    fillcolor='rgba(255, 143, 1, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                # 50% confidence interval
                fig.add_trace(go.Scatter(
                    x=list(forecast_df['date']) + list(forecast_df['date'][::-1]),
                    y=confidence_intervals['q75'] + confidence_intervals['q25'][::-1],
                    fill='toself',
                    fillcolor='rgba(255, 143, 1, 0.4)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='50% Confidence Interval',
                    showlegend=True,
                    hoverinfo='skip'
                ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines+markers',
                name='AI Forecast',
                line=dict(color=COLORS['warning'], width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='<b>Forecast</b><br>%{y:.0f} units<br>%{x}<extra></extra>'
            ))
        
        # Add current date line
        fig.add_vline(
            x=datetime.now(),
            line_dash="solid",
            line_color=COLORS['danger'],
            line_width=2,
            annotation_text="Today",
            annotation_position="top"
        )
        
        # Update layout
        layout = ChartFactory.get_base_layout("AI-Powered Demand Forecasting with Uncertainty", 500)
        layout.update({
            'xaxis_title': 'Date',
            'yaxis_title': 'Demand (Units)',
            'hovermode': 'x unified'
        })
        fig.update_layout(**layout)
        
        return ChartFactory.style_axes(fig)
    
    @staticmethod
    def create_forecast_accuracy_chart(
        accuracy_data: List[Dict[str, Any]]
    ) -> go.Figure:
        """Create forecast accuracy tracking chart"""
        
        if not accuracy_data:
            return go.Figure()
        
        df = pd.DataFrame(accuracy_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAPE Over Time', 'RMSE Trend', 'Bias Analysis', 'Accuracy by Product'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # MAPE trend
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['mape'],
                mode='lines+markers',
                name='MAPE (%)',
                line=dict(color=COLORS['primary'], width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # RMSE trend
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['rmse'],
                mode='lines+markers',
                name='RMSE',
                line=dict(color=COLORS['secondary'], width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # Bias analysis
        fig.add_trace(
            go.Bar(
                x=df['product_category'],
                y=df['bias'],
                name='Forecast Bias',
                marker_color=COLORS['info']
            ),
            row=2, col=1
        )
        
        # Accuracy by product
        fig.add_trace(
            go.Scatter(
                x=df['product_category'],
                y=df['accuracy'],
                mode='markers',
                name='Accuracy Score',
                marker=dict(
                    size=df['volume'],
                    sizemode='diameter',
                    sizeref=2.*max(df['volume'])/(40.**2),
                    color=df['accuracy'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Accuracy", x=1.02)
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        layout = ChartFactory.get_base_layout("Forecast Model Performance Analytics", 600)
        fig.update_layout(**layout)
        
        return ChartFactory.style_axes(fig)

class QualityCharts:
    """Quality assessment chart components"""
    
    @staticmethod
    def create_quality_distribution_chart(
        quality_data: List[Dict[str, Any]]
    ) -> go.Figure:
        """Create comprehensive quality assessment visualization"""
        
        # Quality distribution pie chart
        quality_labels = ['Fresh', 'Good', 'Fair', 'Poor', 'Spoiled']
        quality_counts = [45, 30, 15, 8, 2]
        quality_colors = [COLORS['success'], COLORS['info'], COLORS['warning'], '#fd7e14', COLORS['danger']]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Quality Distribution', 'Quality Trends Over Time',
                'Quality by Product Category', 'Confidence Score Distribution'
            ),
            specs=[
                [{"type": "pie"}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Quality distribution donut chart
        fig.add_trace(
            go.Pie(
                labels=quality_labels,
                values=quality_counts,
                hole=0.4,
                marker_colors=quality_colors,
                textinfo='label+percent',
                textposition='outside',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Quality trends over time
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=7, freq='D')
        for i, label in enumerate(quality_labels):
            trend_data = np.random.poisson(quality_counts[i]/7, 7)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=trend_data,
                    mode='lines+markers',
                    name=label,
                    line=dict(color=quality_colors[i], width=2),
                    marker=dict(size=4)
                ),
                row=1, col=2
            )
        
        # Quality by category
        categories = ['Fruits', 'Vegetables', 'Dairy', 'Meat']
        avg_quality = [0.85, 0.78, 0.92, 0.88]
        fig.add_trace(
            go.Bar(
                x=categories,
                y=avg_quality,
                name='Average Quality Score',
                marker_color=COLORS['gradient_colors'][:4],
                text=[f'{v:.2f}' for v in avg_quality],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Confidence distribution
        confidence_scores = np.random.beta(8, 2, 1000)  # Skewed towards high confidence
        fig.add_trace(
            go.Histogram(
                x=confidence_scores,
                nbinsx=20,
                name='Confidence Distribution',
                marker_color=COLORS['primary'],
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # Update layout
        layout = ChartFactory.get_base_layout("Computer Vision Quality Assessment Analytics", 700)
        fig.update_layout(**layout)
        
        return ChartFactory.style_axes(fig)

class NetworkCharts:
    """Supply chain network visualization components"""
    
    @staticmethod
    def create_network_optimization_chart(
        warehouse_data: Dict[str, Dict[str, Any]],
        route_data: List[Tuple[str, str]]
    ) -> go.Figure:
        """Create interactive supply chain network map"""
        
        fig = go.Figure()
        
        # Add warehouse locations
        for name, data in warehouse_data.items():
            efficiency_color = 'green' if data['efficiency'] > 0.9 else 'orange' if data['efficiency'] > 0.8 else 'red'
            
            fig.add_trace(go.Scattermapbox(
                lat=[data['lat']],
                lon=[data['lon']],
                mode='markers+text',
                marker=dict(
                    size=data['size']/3,
                    color=data['efficiency'],
                    colorscale='RdYlGn',
                    cmin=0.7,
                    cmax=1.0,
                    showscale=True,
                    colorbar=dict(title="Efficiency Score", x=1.02),
                    line=dict(width=2, color='white')
                ),
                text=name,
                textposition='top center',
                textfont=dict(size=12, color='black'),
                hovertemplate=f'<b>{name}</b><br>Efficiency: {data["efficiency"]:.1%}<br>Capacity: {data["size"]} units<extra></extra>',
                name=name
            ))
        
        # Add route connections
        for start, end in route_data:
            if start in warehouse_data and end in warehouse_data:
                fig.add_trace(go.Scattermapbox(
                    lat=[warehouse_data[start]['lat'], warehouse_data[end]['lat']],
                    lon=[warehouse_data[start]['lon'], warehouse_data[end]['lon']],
                    mode='lines',
                    line=dict(width=4, color='rgba(46, 134, 171, 0.8)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Update layout with map configuration
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=63.0, lon=10.0),
                zoom=4.5
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            title={
                'text': 'Supply Chain Network Optimization',
                'x': 0.5,
                'xanchor': 'center'
            },
            showlegend=False
        )
        
        return fig

class SustainabilityCharts:
    """Sustainability and environmental impact charts"""
    
    @staticmethod
    def create_sustainability_dashboard(
        sustainability_data: Dict[str, Any]
    ) -> go.Figure:
        """Create comprehensive sustainability impact visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Environmental Impact Reduction',
                'Carbon Footprint Trend',
                'Waste Reduction by Category',
                'Sustainability Score Progress'
            ),
            specs=[
                [{"type": "bar"}, {"secondary_y": False}],
                [{"type": "pie"}, {"type": "indicator"}]
            ]
        )
        
        # Environmental impact metrics
        metrics = ['CO2 Reduction', 'Waste Reduction', 'Energy Savings', 'Water Conservation']
        current = [25, 30, 18, 22]
        target = [30, 35, 25, 28]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=current,
                name='Current',
                marker_color=COLORS['success'],
                text=[f'{v}%' for v in current],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=target,
                name='Target',
                marker_color='rgba(40, 167, 69, 0.3)',
                text=[f'{v}%' for v in target],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Carbon footprint trend
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
        carbon_trend = 100 - np.cumsum(np.random.exponential(0.5, 30))
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=carbon_trend,
                mode='lines+markers',
                name='Carbon Footprint Reduction',
                line=dict(color=COLORS['success'], width=3),
                fill='tonexty'
            ),
            row=1, col=2
        )
        
        # Waste reduction by category
        waste_categories = ['Food Waste', 'Packaging', 'Transport', 'Energy']
        waste_reduction = [35, 20, 15, 12]
        
        fig.add_trace(
            go.Pie(
                labels=waste_categories,
                values=waste_reduction,
                hole=0.3,
                marker_colors=COLORS['gradient_colors'][:4],
                textinfo='label+percent'
            ),
            row=2, col=1
        )
        
        # Sustainability score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=85,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sustainability Score"},
                delta={'reference': 75},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': COLORS['success']},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        layout = ChartFactory.get_base_layout("Sustainability Impact Dashboard", 700)
        fig.update_layout(**layout)
        
        return fig

class BusinessImpactCharts:
    """Business impact and ROI visualization components"""
    
    @staticmethod
    def create_roi_dashboard(
        financial_data: Dict[str, Any]
    ) -> go.Figure:
        """Create business impact and ROI visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cost Savings Over Time',
                'ROI by Initiative',
                'Revenue Impact',
                'Efficiency Gains'
            )
        )
        
        # Cost savings trend
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        cost_savings = [50000, 75000, 120000, 180000, 220000, 280000]
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=cost_savings,
                mode='lines+markers',
                name='Cumulative Savings',
                line=dict(color=COLORS['success'], width=4),
                marker=dict(size=8),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # ROI by initiative
        initiatives = ['AI Quality Control', 'Demand Forecasting', 'Route Optimization', 'Waste Reduction']
        roi_values = [320, 280, 450, 380]
        
        fig.add_trace(
            go.Bar(
                x=initiatives,
                y=roi_values,
                name='ROI (%)',
                marker_color=COLORS['gradient_colors'][:4],
                text=[f'{v}%' for v in roi_values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Revenue impact
        revenue_categories = ['Reduced Waste', 'Improved Quality', 'Optimized Routes', 'Better Forecasting']
        revenue_impact = [150000, 200000, 120000, 180000]
        
        fig.add_trace(
            go.Waterfall(
                x=revenue_categories,
                y=revenue_impact,
                name='Revenue Impact',
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": COLORS['success']}},
                decreasing={"marker": {"color": COLORS['danger']}},
                totals={"marker": {"color": COLORS['primary']}}
            ),
            row=2, col=1
        )
        
        # Efficiency gains radar chart
        categories = ['Quality<br>Control', 'Inventory<br>Management', 'Route<br>Planning', 
                     'Demand<br>Accuracy', 'Waste<br>Reduction']
        efficiency_scores = [85, 78, 92, 88, 95]
        
        fig.add_trace(
            go.Scatterpolar(
                r=efficiency_scores,
                theta=categories,
                fill='toself',
                name='Efficiency Score',
                line_color=COLORS['primary']
            ),
            row=2, col=2
        )
        
        # Update layout
        layout = ChartFactory.get_base_layout("Business Impact & ROI Dashboard", 700)
        fig.update_layout(**layout)
        
        return fig