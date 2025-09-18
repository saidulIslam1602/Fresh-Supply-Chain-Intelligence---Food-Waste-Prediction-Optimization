#!/usr/bin/env python3
"""
Initialize SQL Server database with real schema and sample data
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
            'DATABASE=master;'
            'UID=sa;'
            'PWD=Saidul1602;'
            'Trusted_Connection=no;'
        )
        return conn
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return None

def create_database():
    """Create FreshSupplyChain database"""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'FreshSupplyChain')
            BEGIN
                CREATE DATABASE FreshSupplyChain;
                PRINT 'Database FreshSupplyChain created successfully';
            END
            ELSE
                PRINT 'Database FreshSupplyChain already exists';
        """)
        conn.commit()
        logger.info("Database created successfully")
        return True
    except Exception as e:
        logger.error(f"Database creation failed: {e}")
        return False
    finally:
        conn.close()

def create_schema():
    """Create database schema"""
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=FreshSupplyChain;'
        'UID=sa;'
        'PWD=Saidul1602;'
        'Trusted_Connection=no;'
    )
    
    try:
        cursor = conn.cursor()
        
        # Read and execute schema
        with open('data/sql_server_schema.sql', 'r') as f:
            schema_sql = f.read()
        
        # Split by GO statements and execute
        statements = schema_sql.split('GO')
        for statement in statements:
            statement = statement.strip()
            if statement:
                try:
                    cursor.execute(statement)
                except Exception as e:
                    logger.warning(f"Statement failed: {e}")
        
        conn.commit()
        logger.info("Schema created successfully")
        return True
    except Exception as e:
        logger.error(f"Schema creation failed: {e}")
        return False
    finally:
        conn.close()

def load_sample_data():
    """Load sample data into database"""
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=FreshSupplyChain;'
        'UID=sa;'
        'PWD=Saidul1602;'
        'Trusted_Connection=no;'
    )
    
    try:
        cursor = conn.cursor()
        
        # Load products
        products = [
            ('PROD_001', 'Fresh Apples', 'Fruits', 'Red Apples', 14, 0, 4, 85, 95, 2.50, 4.99),
            ('PROD_002', 'Organic Bananas', 'Fruits', 'Yellow Bananas', 7, 2, 6, 80, 90, 1.80, 3.49),
            ('PROD_003', 'Fresh Lettuce', 'Vegetables', 'Green Lettuce', 5, 0, 2, 90, 95, 1.20, 2.99),
            ('PROD_004', 'Ripe Tomatoes', 'Vegetables', 'Red Tomatoes', 10, 4, 8, 85, 90, 2.00, 4.49),
            ('PROD_005', 'Fresh Carrots', 'Vegetables', 'Orange Carrots', 21, 0, 4, 90, 95, 1.50, 3.29),
            ('PROD_006', 'Strawberries', 'Fruits', 'Red Strawberries', 5, 0, 2, 90, 95, 3.00, 5.99),
            ('PROD_007', 'Spinach', 'Vegetables', 'Green Spinach', 7, 0, 4, 90, 95, 2.20, 4.49),
            ('PROD_008', 'Oranges', 'Fruits', 'Navel Oranges', 21, 2, 8, 85, 90, 2.80, 4.99),
            ('PROD_009', 'Cucumbers', 'Vegetables', 'Green Cucumbers', 10, 4, 8, 85, 90, 1.80, 3.49),
            ('PROD_010', 'Avocados', 'Fruits', 'Hass Avocados', 7, 4, 8, 85, 90, 2.50, 4.99)
        ]
        
        for product in products:
            cursor.execute("""
                INSERT INTO Products (ProductCode, ProductName, Category, Subcategory, 
                                   ShelfLifeDays, OptimalTempMin, OptimalTempMax, 
                                   OptimalHumidityMin, OptimalHumidityMax, UnitCost, UnitPrice)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, product)
        
        # Load warehouses
        warehouses = [
            ('OSL_01', 'Oslo Central Distribution', 59.9139, 10.7522, 50000, 1, 'Norway', 'Eastern'),
            ('BGO_01', 'Bergen Fresh Hub', 60.3913, 5.3221, 30000, 1, 'Norway', 'Western'),
            ('TRD_01', 'Trondheim Cold Storage', 63.4305, 10.3951, 25000, 1, 'Norway', 'Central'),
            ('STO_01', 'Stockholm Import Center', 59.3293, 18.0686, 40000, 1, 'Sweden', 'Nordic'),
            ('CPH_01', 'Copenhagen Distribution', 55.6761, 12.5683, 35000, 1, 'Denmark', 'Nordic')
        ]
        
        for warehouse in warehouses:
            cursor.execute("""
                INSERT INTO Warehouses (WarehouseCode, WarehouseName, LocationLat, LocationLon,
                                      CapacityUnits, TemperatureControlled, Country, Region)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, warehouse)
        
        # Load inventory
        inventory_data = []
        for product_id in range(1, 11):
            for warehouse_id in range(1, 6):
                for lot in range(1, 4):  # 3 lots per product per warehouse
                    lot_number = f"LOT_{product_id:03d}_{warehouse_id}_{lot:03d}"
                    quantity = np.random.randint(50, 200)
                    production_date = datetime.now() - timedelta(days=np.random.randint(1, 10))
                    expiry_date = production_date + timedelta(days=np.random.randint(5, 20))
                    
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
        
        # Load temperature logs
        temp_data = []
        for warehouse_id in range(1, 6):
            for device in range(1, 6):  # 5 devices per warehouse
                device_id = f"SENSOR_{warehouse_id}_{device:02d}"
                for hour in range(24 * 7):  # 7 days of data
                    log_time = datetime.now() - timedelta(hours=hour)
                    temp = np.random.normal(4, 1.5)
                    humidity = np.random.normal(90, 5)
                    
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
        
        # Load waste events
        waste_data = []
        for day in range(30):  # 30 days of waste data
            for _ in range(np.random.randint(5, 15)):  # 5-15 waste events per day
                product_id = np.random.randint(1, 11)
                warehouse_id = np.random.randint(1, 6)
                lot_number = f"LOT_{product_id:03d}_{warehouse_id}_{np.random.randint(1, 4):03d}"
                quantity = np.random.randint(1, 20)
                waste_date = datetime.now() - timedelta(days=day)
                
                waste_data.append((
                    product_id, warehouse_id, lot_number, quantity,
                    'Expired', 'Quality', quantity * np.random.uniform(1, 5),
                    waste_date, np.random.choice([0, 1]), np.random.randint(0, 5)
                ))
        
        for waste in waste_data:
            cursor.execute("""
                INSERT INTO WasteEvents (ProductID, WarehouseID, LotNumber, QuantityWasted,
                                       WasteReason, WasteCategory, EstimatedValueLoss,
                                       RecordedAt, TemperatureViolation, DaysPastOptimal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, waste)
        
        conn.commit()
        logger.info("Sample data loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return False
    finally:
        conn.close()

def main():
    """Main initialization function"""
    logger.info("Starting database initialization...")
    
    if create_database():
        logger.info("Database created")
    else:
        logger.error("Database creation failed")
        return
    
    if create_schema():
        logger.info("Schema created")
    else:
        logger.error("Schema creation failed")
        return
    
    if load_sample_data():
        logger.info("Sample data loaded")
    else:
        logger.error("Data loading failed")
        return
    
    logger.info("Database initialization completed successfully!")

if __name__ == "__main__":
    main()