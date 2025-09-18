"""
Graph Neural Network for supply chain optimization
Models the supply chain as a graph where nodes are locations and edges are routes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)

class SupplyChainGNN(nn.Module):
    """
    Graph Neural Network for supply chain optimization
    Models the supply chain as a graph where nodes are locations and edges are routes
    """
    
    def __init__(self, input_features: int, hidden_size: int = 128, output_size: int = 32):
        super().__init__()
        
        self.conv1 = GCNConv(input_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.conv3 = GCNConv(hidden_size, output_size)
        
        self.dropout = nn.Dropout(0.2)
        
        # Edge prediction layers
        self.edge_predictor = nn.Sequential(
            nn.Linear(output_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, batch=None):
        """Forward pass through GNN"""
        
        # Node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # Global graph representation
        if batch is not None:
            x_global = global_mean_pool(x, batch)
        else:
            x_global = x.mean(dim=0, keepdim=True)
        
        return x, x_global
    
    def predict_edge_importance(self, x, edge_index):
        """Predict importance of edges in supply chain"""
        
        edge_features = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            edge_feat = torch.cat([x[src], x[dst]])
            edge_features.append(edge_feat)
        
        edge_features = torch.stack(edge_features)
        edge_importance = self.edge_predictor(edge_features)
        
        return edge_importance


class SupplyChainOptimizer:
    """
    Optimization engine using Gurobi for supply chain decisions
    Integrates with GNN for intelligent routing
    """
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        self.gnn_model = SupplyChainGNN(input_features=10)
        self.graph = None
        
    def build_supply_network(self):
        """Build supply chain network from database"""
        
        # Load nodes
        nodes_query = "SELECT * FROM SupplyChainNodes"
        nodes_df = pd.read_sql(nodes_query, self.engine)
        
        # Load edges
        edges_query = "SELECT * FROM SupplyChainEdges"
        edges_df = pd.read_sql(edges_query, self.engine)
        
        # Create NetworkX graph
        self.graph = nx.DiGraph()
        
        for _, node in nodes_df.iterrows():
            self.graph.add_node(
                node['NodeID'],
                type=node['NodeType'],
                name=node['NodeName'],
                capacity=node['Capacity'],
                lat=node['LocationLat'],
                lon=node['LocationLon']
            )
        
        for _, edge in edges_df.iterrows():
            self.graph.add_edge(
                edge['SourceNodeID'],
                edge['TargetNodeID'],
                distance=edge['DistanceKM'],
                transit_time=edge['TransitTimeDays'],
                cost=edge['CostPerUnit']
            )
        
        return self.graph
    
    def optimize_distribution(self, demand: Dict, inventory: Dict, shelf_life: Dict):
        """
        Optimize distribution using Gurobi
        Minimizes cost while respecting shelf life constraints
        """
        
        model = gp.Model("FreshProduceDistribution")
        
        # Decision variables
        flow = {}
        for edge in self.graph.edges():
            for product in demand.keys():
                flow[edge, product] = model.addVar(
                    lb=0, 
                    vtype=GRB.CONTINUOUS,
                    name=f"flow_{edge[0]}_{edge[1]}_{product}"
                )
        
        waste = {}
        for node in self.graph.nodes():
            for product in demand.keys():
                waste[node, product] = model.addVar(
                    lb=0,
                    vtype=GRB.CONTINUOUS,
                    name=f"waste_{node}_{product}"
                )
        
        # Objective: Minimize total cost (transportation + waste)
        transport_cost = gp.quicksum(
            flow[edge, product] * self.graph[edge[0]][edge[1]]['cost']
            for edge in self.graph.edges()
            for product in demand.keys()
        )
        
        waste_cost = gp.quicksum(
            waste[node, product] * 10  # Penalty for waste
            for node in self.graph.nodes()
            for product in demand.keys()
        )
        
        model.setObjective(transport_cost + waste_cost, GRB.MINIMIZE)
        
        # Constraints
        
        # 1. Flow conservation
        for node in self.graph.nodes():
            for product in demand.keys():
                inflow = gp.quicksum(
                    flow[edge, product]
                    for edge in self.graph.in_edges(node)
                )
                outflow = gp.quicksum(
                    flow[edge, product]
                    for edge in self.graph.out_edges(node)
                )
                
                supply = inventory.get((node, product), 0)
                node_demand = demand.get((node, product), 0)
                
                model.addConstr(
                    inflow + supply == outflow + node_demand + waste[node, product],
                    name=f"flow_conservation_{node}_{product}"
                )
        
        # 2. Capacity constraints
        for node in self.graph.nodes():
            total_flow = gp.quicksum(
                flow[edge, product]
                for edge in self.graph.out_edges(node)
                for product in demand.keys()
            )
            
            capacity = self.graph.nodes[node]['capacity']
            model.addConstr(
                total_flow <= capacity,
                name=f"capacity_{node}"
            )
        
        # 3. Shelf life constraints
        for edge in self.graph.edges():
            transit_time = self.graph[edge[0]][edge[1]]['transit_time']
            
            for product in demand.keys():
                remaining_shelf = shelf_life.get(product, 30)
                
                if transit_time > remaining_shelf * 0.5:  # Don't ship if >50% shelf life used
                    model.addConstr(
                        flow[edge, product] == 0,
                        name=f"shelf_life_{edge[0]}_{edge[1]}_{product}"
                    )
        
        # Solve
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            solution = {
                'objective': model.objVal,
                'flows': {
                    (edge, product): flow[edge, product].x
                    for edge in self.graph.edges()
                    for product in demand.keys()
                },
                'waste': {
                    (node, product): waste[node, product].x
                    for node in self.graph.nodes()
                    for product in demand.keys()
                }
            }
            return solution
        else:
            logger.error("Optimization failed")
            return None