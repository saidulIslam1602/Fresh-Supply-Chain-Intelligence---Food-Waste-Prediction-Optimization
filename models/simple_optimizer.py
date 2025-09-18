"""
Simplified optimization model without Gurobi dependency
Uses NetworkX and scipy for basic optimization
"""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleSupplyChainOptimizer:
    """
    Simplified supply chain optimizer using NetworkX and scipy
    Replaces Gurobi-based optimization for demo purposes
    """
    
    def __init__(self, connection_string: str = None):
        self.connection_string = connection_string
        self.graph = None
        
    def build_supply_network(self):
        """Build supply chain network from mock data"""
        
        # Create a simple network for Norway
        self.graph = nx.DiGraph()
        
        # Add nodes (warehouses and stores)
        nodes = [
            (1, {'type': 'SUPPLIER', 'name': 'Nordic Fresh Suppliers', 'lat': 59.3293, 'lon': 18.0686, 'capacity': 10000}),
            (2, {'type': 'WAREHOUSE', 'name': 'Oslo Central Distribution', 'lat': 59.9139, 'lon': 10.7522, 'capacity': 50000}),
            (3, {'type': 'WAREHOUSE', 'name': 'Bergen Fresh Hub', 'lat': 60.3913, 'lon': 5.3221, 'capacity': 30000}),
            (4, {'type': 'WAREHOUSE', 'name': 'Trondheim Cold Storage', 'lat': 63.4305, 'lon': 10.3951, 'capacity': 25000}),
            (5, {'type': 'RETAIL', 'name': 'Oslo City Center Store', 'lat': 59.9139, 'lon': 10.7522, 'capacity': 1000}),
            (6, {'type': 'RETAIL', 'name': 'Bergen Market Store', 'lat': 60.3913, 'lon': 5.3221, 'capacity': 800}),
            (7, {'type': 'RETAIL', 'name': 'Trondheim Fresh Store', 'lat': 63.4305, 'lon': 10.3951, 'capacity': 600})
        ]
        
        self.graph.add_nodes_from(nodes)
        
        # Add edges (transportation routes)
        edges = [
            (1, 2, {'distance': 500, 'transit_time': 1.0, 'cost': 0.5}),
            (1, 3, {'distance': 600, 'transit_time': 1.2, 'cost': 0.6}),
            (1, 4, {'distance': 800, 'transit_time': 1.5, 'cost': 0.8}),
            (2, 3, {'distance': 300, 'transit_time': 0.6, 'cost': 0.3}),
            (2, 4, {'distance': 400, 'transit_time': 0.8, 'cost': 0.4}),
            (3, 4, {'distance': 350, 'transit_time': 0.7, 'cost': 0.35}),
            (2, 5, {'distance': 10, 'transit_time': 0.1, 'cost': 0.05}),
            (3, 6, {'distance': 15, 'transit_time': 0.2, 'cost': 0.08}),
            (4, 7, {'distance': 12, 'transit_time': 0.15, 'cost': 0.06}),
            (2, 6, {'distance': 300, 'transit_time': 0.6, 'cost': 0.3}),
            (2, 7, {'distance': 400, 'transit_time': 0.8, 'cost': 0.4}),
            (3, 5, {'distance': 300, 'transit_time': 0.6, 'cost': 0.3}),
            (3, 7, {'distance': 350, 'transit_time': 0.7, 'cost': 0.35}),
            (4, 5, {'distance': 400, 'transit_time': 0.8, 'cost': 0.4}),
            (4, 6, {'distance': 350, 'transit_time': 0.7, 'cost': 0.35})
        ]
        
        self.graph.add_edges_from(edges)
        
        logger.info(f"Built supply network with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def optimize_distribution(self, demand: Dict, inventory: Dict, shelf_life: Dict):
        """
        Optimize distribution using simple linear programming
        """
        
        if self.graph is None:
            self.build_supply_network()
        
        # Convert to simple optimization problem
        # For demo purposes, we'll use a simplified approach
        
        # Calculate shortest paths for each demand
        optimal_routes = {}
        total_cost = 0
        
        for (warehouse_id, product_id), demand_qty in demand.items():
            if demand_qty <= 0:
                continue
                
            # Find shortest path from suppliers to this warehouse
            suppliers = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'SUPPLIER']
            warehouses = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'WAREHOUSE']
            
            if suppliers and warehouses:
                # Simple routing: use the first available supplier
                supplier = suppliers[0]
                target_warehouse = warehouse_id if warehouse_id in warehouses else warehouses[0]
                
                try:
                    path = nx.shortest_path(self.graph, supplier, target_warehouse, weight='cost')
                    
                    # Calculate total cost for this path
                    path_cost = 0
                    for i in range(len(path) - 1):
                        edge_cost = self.graph[path[i]][path[i+1]]['cost']
                        path_cost += edge_cost
                    
                    total_cost += path_cost * demand_qty
                    
                    # Store route
                    route_key = f"supplier_{supplier}_to_warehouse_{target_warehouse}_product_{product_id}"
                    optimal_routes[route_key] = {
                        'path': path,
                        'quantity': demand_qty,
                        'cost_per_unit': path_cost,
                        'total_cost': path_cost * demand_qty
                    }
                    
                except nx.NetworkXNoPath:
                    logger.warning(f"No path found from supplier to warehouse {target_warehouse}")
        
        # Calculate waste (simplified)
        waste = {}
        for (warehouse_id, product_id), inv_qty in inventory.items():
            if inv_qty > 0:
                # Simple waste calculation based on shelf life
                remaining_shelf = shelf_life.get(product_id, 30)
                waste_probability = max(0, (30 - remaining_shelf) / 30)
                waste_qty = inv_qty * waste_probability * 0.1  # 10% of probability
                waste[(warehouse_id, product_id)] = waste_qty
        
        solution = {
            'objective': total_cost + sum(waste.values()) * 10,  # Add waste penalty
            'flows': optimal_routes,
            'waste': waste,
            'total_transport_cost': total_cost,
            'total_waste_cost': sum(waste.values()) * 10
        }
        
        logger.info(f"Optimization completed. Total cost: {solution['objective']:.2f}")
        return solution
    
    def get_network_metrics(self):
        """Get network analysis metrics"""
        if self.graph is None:
            self.build_supply_network()
        
        metrics = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'average_clustering': nx.average_clustering(self.graph.to_undirected())
        }
        
        return metrics