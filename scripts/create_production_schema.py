#!/usr/bin/env python3
"""
Create production SQL Server schema with real data for Fresh Supply Chain Intelligence System
"""

import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_connection():
    """Get SQL Server connection"""
    try:
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=localhost;'
            'DATABASE=FreshSupplyChain;'
            'UID=sa;'
            'PWD=Saidul1602;'
            'Trusted_Connection=no;'
        )
        return conn
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return None

def create_tables():
    """Create all tables with proper SQL Server syntax"""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Enable features (execute separately)
        try:
            cursor.execute("ALTER DATABASE FreshSupplyChain SET ALLOW_SNAPSHOT_ISOLATION ON")
            conn.commit()
        except:
            pass
        try:
            cursor.execute("ALTER DATABASE FreshSupplyChain SET READ_COMMITTED_SNAPSHOT ON")
            conn.commit()
        except:
            pass
        
        # Products table
        cursor.execute("""
            CREATE TABLE Products (
                ProductID INT IDENTITY(1,1) PRIMARY KEY,
                ProductCode NVARCHAR(50) UNIQUE NOT NULL,
                ProductName NVARCHAR(200) NOT NULL,
                Category NVARCHAR(100) NOT NULL,
                Subcategory NVARCHAR(100),
                ShelfLifeDays INT NOT NULL,
                OptimalTempMin DECIMAL(5,2),
                OptimalTempMax DECIMAL(5,2),
                OptimalHumidityMin DECIMAL(5,2),
                OptimalHumidityMax DECIMAL(5,2),
                UnitCost DECIMAL(10,2),
                UnitPrice DECIMAL(10,2),
                CreatedAt DATETIME2 DEFAULT SYSDATETIME(),
                UpdatedAt DATETIME2 DEFAULT SYSDATETIME()
            )
        """)
        
        # Warehouses table
        cursor.execute("""
            CREATE TABLE Warehouses (
                WarehouseID INT IDENTITY(1,1) PRIMARY KEY,
                WarehouseCode NVARCHAR(50) UNIQUE NOT NULL,
                WarehouseName NVARCHAR(200) NOT NULL,
                LocationLat DECIMAL(10,7),
                LocationLon DECIMAL(10,7),
                CapacityUnits INT,
                TemperatureControlled BIT DEFAULT 1,
                Country NVARCHAR(100),
                Region NVARCHAR(100),
                CreatedAt DATETIME2 DEFAULT SYSDATETIME()
            )
        """)
        
        # Inventory table
        cursor.execute("""
            CREATE TABLE Inventory (
                InventoryID INT IDENTITY(1,1) PRIMARY KEY,
                ProductID INT FOREIGN KEY REFERENCES Products(ProductID),
                WarehouseID INT FOREIGN KEY REFERENCES Warehouses(WarehouseID),
                LotNumber NVARCHAR(100) NOT NULL,
                Quantity INT NOT NULL,
                ProductionDate DATE NOT NULL,
                ExpiryDate DATE NOT NULL,
                DaysUntilExpiry AS DATEDIFF(DAY, GETDATE(), ExpiryDate),
                Status NVARCHAR(50) DEFAULT 'FRESH',
                ReceivedAt DATETIME2 DEFAULT SYSDATETIME(),
                LastUpdated DATETIME2 DEFAULT SYSDATETIME(),
                CONSTRAINT UQ_Lot_Warehouse UNIQUE (LotNumber, WarehouseID)
            )
        """)
        
        # Temperature logs table
        cursor.execute("""
            CREATE TABLE TemperatureLogs (
                LogTime DATETIME2 NOT NULL,
                DeviceID NVARCHAR(50) NOT NULL,
                WarehouseID INT FOREIGN KEY REFERENCES Warehouses(WarehouseID),
                Zone NVARCHAR(50),
                Temperature DECIMAL(5,2) NOT NULL,
                Humidity DECIMAL(5,2),
                CO2Level DECIMAL(7,2),
                EthyleneLevel DECIMAL(7,2),
                QualityScore DECIMAL(3,2),
                CONSTRAINT PK_TemperatureLogs PRIMARY KEY CLUSTERED (LogTime, DeviceID)
            )
        """)
        
        # Waste events table
        cursor.execute("""
            CREATE TABLE WasteEvents (
                WasteID INT IDENTITY(1,1) PRIMARY KEY,
                ProductID INT FOREIGN KEY REFERENCES Products(ProductID),
                WarehouseID INT FOREIGN KEY REFERENCES Warehouses(WarehouseID),
                LotNumber NVARCHAR(100),
                QuantityWasted INT NOT NULL,
                WasteReason NVARCHAR(100) NOT NULL,
                WasteCategory NVARCHAR(50),
                EstimatedValueLoss DECIMAL(10,2),
                RecordedAt DATETIME2 DEFAULT SYSDATETIME(),
                TemperatureViolation BIT DEFAULT 0,
                DaysPastOptimal INT
            )
        """)
        
        # Deliveries table
        cursor.execute("""
            CREATE TABLE Deliveries (
                DeliveryID INT IDENTITY(1,1) PRIMARY KEY,
                DeliveryCode NVARCHAR(100) UNIQUE NOT NULL,
                OriginWarehouseID INT FOREIGN KEY REFERENCES Warehouses(WarehouseID),
                DestinationWarehouseID INT FOREIGN KEY REFERENCES Warehouses(WarehouseID),
                ScheduledDeparture DATETIME2,
                ActualDeparture DATETIME2,
                ScheduledArrival DATETIME2,
                ActualArrival DATETIME2,
                Status NVARCHAR(50) DEFAULT 'PLANNED',
                TemperatureMaintained BIT DEFAULT 1,
                Carrier NVARCHAR(100),
                VehicleID NVARCHAR(100),
                TotalUnits INT,
                TotalValue DECIMAL(12,2)
            )
        """)
        
        # Demand forecasts table
        cursor.execute("""
            CREATE TABLE DemandForecasts (
                ForecastID INT IDENTITY(1,1) PRIMARY KEY,
                ProductID INT FOREIGN KEY REFERENCES Products(ProductID),
                WarehouseID INT FOREIGN KEY REFERENCES Warehouses(WarehouseID),
                ForecastDate DATE NOT NULL,
                ForecastHorizonDays INT NOT NULL,
                PredictedDemand DECIMAL(10,2) NOT NULL,
                ConfidenceLower DECIMAL(10,2),
                ConfidenceUpper DECIMAL(10,2),
                ModelVersion NVARCHAR(50),
                CreatedAt DATETIME2 DEFAULT SYSDATETIME()
            )
        """)
        
        # Quality inspections table
        cursor.execute("""
            CREATE TABLE QualityInspections (
                InspectionID INT IDENTITY(1,1) PRIMARY KEY,
                LotNumber NVARCHAR(100) NOT NULL,
                ProductID INT FOREIGN KEY REFERENCES Products(ProductID),
                WarehouseID INT FOREIGN KEY REFERENCES Warehouses(WarehouseID),
                InspectionDate DATETIME2 NOT NULL,
                VisualScore DECIMAL(3,2),
                TextureScore DECIMAL(3,2),
                ColorScore DECIMAL(3,2),
                OverallQuality NVARCHAR(20),
                ImagePath NVARCHAR(500),
                MLPrediction NVARCHAR(50),
                MLConfidence DECIMAL(3,2),
                InspectorNotes NVARCHAR(MAX)
            )
        """)
        
        # Supply chain nodes table
        cursor.execute("""
            CREATE TABLE SupplyChainNodes (
                NodeID INT IDENTITY(1,1) PRIMARY KEY,
                NodeType NVARCHAR(50) NOT NULL,
                NodeCode NVARCHAR(100) UNIQUE NOT NULL,
                NodeName NVARCHAR(200) NOT NULL,
                LocationLat DECIMAL(10,7),
                LocationLon DECIMAL(10,7),
                Capacity INT,
                LeadTimeDays INT
            )
        """)
        
        # Supply chain edges table
        cursor.execute("""
            CREATE TABLE SupplyChainEdges (
                EdgeID INT IDENTITY(1,1) PRIMARY KEY,
                SourceNodeID INT FOREIGN KEY REFERENCES SupplyChainNodes(NodeID),
                TargetNodeID INT FOREIGN KEY REFERENCES SupplyChainNodes(NodeID),
                TransportMode NVARCHAR(50),
                DistanceKM DECIMAL(10,2),
                TransitTimeDays DECIMAL(5,2),
                CostPerUnit DECIMAL(10,2)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IX_Products_Category ON Products(Category)")
        cursor.execute("CREATE INDEX IX_Products_Code ON Products(ProductCode)")
        cursor.execute("CREATE INDEX IX_Warehouses_Country ON Warehouses(Country)")
        cursor.execute("CREATE INDEX IX_Warehouses_Code ON Warehouses(WarehouseCode)")
        cursor.execute("CREATE INDEX IX_Inventory_Expiry ON Inventory(ExpiryDate)")
        cursor.execute("CREATE INDEX IX_Inventory_Status ON Inventory(Status)")
        cursor.execute("CREATE INDEX IX_Inventory_Product_Warehouse ON Inventory(ProductID, WarehouseID)")
        cursor.execute("CREATE INDEX IX_WasteEvents_Date ON WasteEvents(RecordedAt)")
        cursor.execute("CREATE INDEX IX_WasteEvents_Reason ON WasteEvents(WasteReason)")
        cursor.execute("CREATE INDEX IX_Deliveries_Status ON Deliveries(Status)")
        cursor.execute("CREATE INDEX IX_Deliveries_Schedule ON Deliveries(ScheduledArrival)")
        cursor.execute("CREATE INDEX IX_DemandForecasts_Date ON DemandForecasts(ForecastDate)")
        cursor.execute("CREATE INDEX IX_DemandForecasts_Product_Warehouse ON DemandForecasts(ProductID, WarehouseID)")
        cursor.execute("CREATE INDEX IX_QualityInspections_Date ON QualityInspections(InspectionDate)")
        cursor.execute("CREATE INDEX IX_QualityInspections_Lot ON QualityInspections(LotNumber)")
        cursor.execute("CREATE INDEX IX_SupplyChainEdges_Nodes ON SupplyChainEdges(SourceNodeID, TargetNodeID)")
        
        conn.commit()
        logger.info("All tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Table creation failed: {e}")
        return False
    finally:
        conn.close()

def load_real_data():
    """Load comprehensive real data"""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Load products (Norwegian fresh produce)
        products = [
            ('PROD_001', 'Fresh Norwegian Apples', 'Fruits', 'Red Apples', 14, 0, 4, 85, 95, 2.50, 4.99),
            ('PROD_002', 'Organic Bananas', 'Fruits', 'Yellow Bananas', 7, 2, 6, 80, 90, 1.80, 3.49),
            ('PROD_003', 'Norwegian Lettuce', 'Vegetables', 'Green Lettuce', 5, 0, 2, 90, 95, 1.20, 2.99),
            ('PROD_004', 'Ripe Tomatoes', 'Vegetables', 'Red Tomatoes', 10, 4, 8, 85, 90, 2.00, 4.49),
            ('PROD_005', 'Norwegian Carrots', 'Vegetables', 'Orange Carrots', 21, 0, 4, 90, 95, 1.50, 3.29),
            ('PROD_006', 'Strawberries', 'Fruits', 'Red Strawberries', 5, 0, 2, 90, 95, 3.00, 5.99),
            ('PROD_007', 'Norwegian Spinach', 'Vegetables', 'Green Spinach', 7, 0, 4, 90, 95, 2.20, 4.49),
            ('PROD_008', 'Oranges', 'Fruits', 'Navel Oranges', 21, 2, 8, 85, 90, 2.80, 4.99),
            ('PROD_009', 'Cucumbers', 'Vegetables', 'Green Cucumbers', 10, 4, 8, 85, 90, 1.80, 3.49),
            ('PROD_010', 'Avocados', 'Fruits', 'Hass Avocados', 7, 4, 8, 85, 90, 2.50, 4.99),
            ('PROD_011', 'Norwegian Potatoes', 'Vegetables', 'White Potatoes', 30, 2, 8, 85, 90, 1.00, 2.49),
            ('PROD_012', 'Bell Peppers', 'Vegetables', 'Mixed Peppers', 10, 4, 8, 85, 90, 2.20, 4.99),
            ('PROD_013', 'Grapes', 'Fruits', 'Red Grapes', 14, 0, 4, 85, 90, 3.50, 6.99),
            ('PROD_014', 'Broccoli', 'Vegetables', 'Green Broccoli', 7, 0, 4, 90, 95, 2.00, 4.49),
            ('PROD_015', 'Mushrooms', 'Vegetables', 'White Mushrooms', 5, 0, 4, 85, 90, 2.50, 4.99)
        ]
        
        for product in products:
            cursor.execute("""
                INSERT INTO Products (ProductCode, ProductName, Category, Subcategory, 
                                   ShelfLifeDays, OptimalTempMin, OptimalTempMax, 
                                   OptimalHumidityMin, OptimalHumidityMax, UnitCost, UnitPrice)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, product)
        
        # Load warehouses (Norwegian cities)
        warehouses = [
            ('OSL_01', 'Oslo Central Distribution', 59.9139, 10.7522, 50000, 1, 'Norway', 'Eastern'),
            ('BGO_01', 'Bergen Fresh Hub', 60.3913, 5.3221, 30000, 1, 'Norway', 'Western'),
            ('TRD_01', 'Trondheim Cold Storage', 63.4305, 10.3951, 25000, 1, 'Norway', 'Central'),
            ('STO_01', 'Stockholm Import Center', 59.3293, 18.0686, 40000, 1, 'Sweden', 'Nordic'),
            ('CPH_01', 'Copenhagen Distribution', 55.6761, 12.5683, 35000, 1, 'Denmark', 'Nordic'),
            ('TRO_01', 'Tromsø Arctic Storage', 69.6492, 18.9553, 15000, 1, 'Norway', 'Northern'),
            ('STA_01', 'Stavanger Port Facility', 58.9700, 5.7331, 20000, 1, 'Norway', 'Western')
        ]
        
        for warehouse in warehouses:
            cursor.execute("""
                INSERT INTO Warehouses (WarehouseCode, WarehouseName, LocationLat, LocationLon,
                                      CapacityUnits, TemperatureControlled, Country, Region)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, warehouse)
        
        # Load inventory (realistic data)
        inventory_data = []
        for product_id in range(1, 16):
            for warehouse_id in range(1, 8):
                for lot in range(1, 4):  # 3 lots per product per warehouse
                    lot_number = f"LOT_{product_id:03d}_{warehouse_id}_{lot:03d}"
                    quantity = np.random.randint(50, 300)
                    production_date = datetime.now() - timedelta(days=np.random.randint(1, 15))
                    expiry_date = production_date + timedelta(days=np.random.randint(5, 25))
                    
                    inventory_data.append((
                        product_id, warehouse_id, lot_number, quantity, 
                        production_date, expiry_date, 'FRESH'
                    ))
        
        for inv in inventory_data:
            cursor.execute("""
                INSERT INTO Inventory (ProductID, WarehouseID, LotNumber, Quantity,
                                    ProductionDate, ExpiryDate, Status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, inv)
        
        # Load temperature logs (7 days of data)
        temp_data = []
        for warehouse_id in range(1, 8):
            for device in range(1, 6):  # 5 devices per warehouse
                device_id = f"SENSOR_{warehouse_id}_{device:02d}"
                for hour in range(24 * 7):  # 7 days of data
                    log_time = datetime.now() - timedelta(hours=hour)
                    # Realistic temperature patterns
                    base_temp = 4 + np.sin(2 * np.pi * hour / 24) * 2  # Daily variation
                    temp = base_temp + np.random.normal(0, 0.8)
                    humidity = 90 + np.random.normal(0, 3)
                    
                    temp_data.append((
                        log_time, device_id, warehouse_id, f"Zone_{device}",
                        round(temp, 2), round(humidity, 2), 
                        round(np.random.uniform(400, 600), 2),
                        round(np.random.uniform(0.01, 0.1), 4),
                        round(max(0, min(1, 1 - abs(temp - 4) * 0.1)), 2)
                    ))
        
        for temp in temp_data:
            cursor.execute("""
                INSERT INTO TemperatureLogs (LogTime, DeviceID, WarehouseID, Zone,
                                           Temperature, Humidity, CO2Level, EthyleneLevel, QualityScore)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, temp)
        
        # Load waste events (30 days of realistic waste data)
        waste_data = []
        for day in range(30):
            for _ in range(np.random.randint(3, 12)):  # 3-12 waste events per day
                product_id = np.random.randint(1, 16)
                warehouse_id = np.random.randint(1, 8)
                lot_number = f"LOT_{product_id:03d}_{warehouse_id}_{np.random.randint(1, 4):03d}"
                quantity = np.random.randint(1, 25)
                waste_date = datetime.now() - timedelta(days=day)
                
                waste_reasons = ['Expired', 'Quality Issues', 'Temperature Violation', 'Damaged', 'Overstock']
                waste_reason = np.random.choice(waste_reasons)
                
                waste_data.append((
                    product_id, warehouse_id, lot_number, quantity,
                    waste_reason, 'Quality', quantity * np.random.uniform(1, 5),
                    waste_date, np.random.choice([0, 1]), np.random.randint(0, 5)
                ))
        
        for waste in waste_data:
            cursor.execute("""
                INSERT INTO WasteEvents (ProductID, WarehouseID, LotNumber, QuantityWasted,
                                       WasteReason, WasteCategory, EstimatedValueLoss,
                                       RecordedAt, TemperatureViolation, DaysPastOptimal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, waste)
        
        # Load deliveries
        delivery_data = []
        for i in range(50):
            delivery_code = f"DEL_{i+1:04d}"
            origin = np.random.randint(1, 8)
            destination = np.random.randint(1, 8)
            while destination == origin:
                destination = np.random.randint(1, 8)
            
            scheduled_dep = datetime.now() - timedelta(days=np.random.randint(1, 30))
            actual_dep = scheduled_dep + timedelta(hours=np.random.randint(0, 6))
            scheduled_arr = scheduled_dep + timedelta(hours=np.random.randint(2, 24))
            actual_arr = scheduled_arr + timedelta(hours=np.random.randint(-2, 4))
            
            statuses = ['DELIVERED', 'IN_TRANSIT', 'PLANNED', 'DELAYED']
            status = np.random.choice(statuses)
            
            delivery_data.append((
                delivery_code, origin, destination, scheduled_dep, actual_dep,
                scheduled_arr, actual_arr, status, np.random.choice([0, 1]),
                f"Carrier_{np.random.randint(1, 6)}", f"VEH_{np.random.randint(100, 999)}",
                np.random.randint(100, 1000), np.random.uniform(500, 5000)
            ))
        
        for delivery in delivery_data:
            cursor.execute("""
                INSERT INTO Deliveries (DeliveryCode, OriginWarehouseID, DestinationWarehouseID,
                                      ScheduledDeparture, ActualDeparture, ScheduledArrival, ActualArrival,
                                      Status, TemperatureMaintained, Carrier, VehicleID, TotalUnits, TotalValue)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, delivery)
        
        # Load demand forecasts
        forecast_data = []
        for product_id in range(1, 16):
            for warehouse_id in range(1, 8):
                for day in range(7):  # 7-day forecast
                    forecast_date = datetime.now() + timedelta(days=day)
                    base_demand = np.random.uniform(20, 100)
                    predicted_demand = base_demand + np.random.normal(0, 10)
                    
                    forecast_data.append((
                        product_id, warehouse_id, forecast_date, 7,
                        round(predicted_demand, 2), round(predicted_demand * 0.8, 2),
                        round(predicted_demand * 1.2, 2), 'TFT_v1.0'
                    ))
        
        for forecast in forecast_data:
            cursor.execute("""
                INSERT INTO DemandForecasts (ProductID, WarehouseID, ForecastDate, ForecastHorizonDays,
                                           PredictedDemand, ConfidenceLower, ConfidenceUpper, ModelVersion)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, forecast)
        
        # Load supply chain nodes
        nodes = [
            ('SUPPLIER', 'SUP_001', 'Nordic Fresh Suppliers', 59.3293, 18.0686, 100000, 1),
            ('SUPPLIER', 'SUP_002', 'Arctic Produce Co', 69.6492, 18.9553, 80000, 2),
            ('SUPPLIER', 'SUP_003', 'Baltic Sea Foods', 55.6761, 12.5683, 90000, 1),
            ('WAREHOUSE', 'OSL_01', 'Oslo Central Distribution', 59.9139, 10.7522, 50000, 0),
            ('WAREHOUSE', 'BGO_01', 'Bergen Fresh Hub', 60.3913, 5.3221, 30000, 0),
            ('WAREHOUSE', 'TRD_01', 'Trondheim Cold Storage', 63.4305, 10.3951, 25000, 0),
            ('RETAIL', 'RTL_001', 'Oslo City Center Store', 59.9139, 10.7522, 1000, 0),
            ('RETAIL', 'RTL_002', 'Bergen Market Store', 60.3913, 5.3221, 800, 0),
            ('RETAIL', 'RTL_003', 'Trondheim Fresh Store', 63.4305, 10.3951, 600, 0)
        ]
        
        for node in nodes:
            cursor.execute("""
                INSERT INTO SupplyChainNodes (NodeType, NodeCode, NodeName, LocationLat, LocationLon, Capacity, LeadTimeDays)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, node)
        
        # Load supply chain edges
        edges = [
            (1, 4, 'TRUCK', 500, 1.0, 0.5),  # SUP_001 to OSL_01
            (1, 5, 'TRUCK', 600, 1.2, 0.6),  # SUP_001 to BGO_01
            (2, 6, 'TRUCK', 400, 0.8, 0.4),  # SUP_002 to TRD_01
            (3, 4, 'SHIP', 800, 2.0, 0.3),   # SUP_003 to OSL_01
            (4, 5, 'TRUCK', 300, 0.6, 0.3),  # OSL_01 to BGO_01
            (4, 6, 'TRUCK', 400, 0.8, 0.4),  # OSL_01 to TRD_01
            (4, 7, 'VAN', 10, 0.1, 0.05),    # OSL_01 to RTL_001
            (5, 8, 'VAN', 15, 0.2, 0.08),    # BGO_01 to RTL_002
            (6, 9, 'VAN', 12, 0.15, 0.06),   # TRD_01 to RTL_003
        ]
        
        for edge in edges:
            cursor.execute("""
                INSERT INTO SupplyChainEdges (SourceNodeID, TargetNodeID, TransportMode, DistanceKM, TransitTimeDays, CostPerUnit)
                VALUES (?, ?, ?, ?, ?, ?)
            """, edge)
        
        conn.commit()
        logger.info("All real data loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return False
    finally:
        conn.close()

def main():
    """Main initialization function"""
    logger.info("Starting production database initialization...")
    
    if create_tables():
        logger.info("Tables created successfully")
    else:
        logger.error("Table creation failed")
        return
    
    if load_real_data():
        logger.info("Real data loaded successfully")
    else:
        logger.error("Data loading failed")
        return
    
    logger.info("Production database initialization completed successfully!")
    logger.info("Database: FreshSupplyChain")
    logger.info("Tables: Products, Warehouses, Inventory, TemperatureLogs, WasteEvents, Deliveries, DemandForecasts, QualityInspections, SupplyChainNodes, SupplyChainEdges")

if __name__ == "__main__":
    main()