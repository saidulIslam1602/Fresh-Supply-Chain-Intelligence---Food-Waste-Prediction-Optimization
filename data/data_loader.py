"""
Data loader for Fresh Supply Chain Intelligence System
Handles data ingestion from multiple sources including USDA, synthetic IoT data, and Fruits-360 dataset
"""

import pandas as pd
import numpy as np
import pyodbc
from sqlalchemy import create_engine, text
import requests
import zipfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)

class FreshSupplyDataLoader:
    """Data loader for Fresh Supply Chain Intelligence System"""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        self.connection_string = connection_string
        
    def download_usda_data(self, output_dir: str = './data/raw'):
        """Download USDA FoodData Central dataset"""
        os.makedirs(output_dir, exist_ok=True)
        
        # USDA FoodData Central download URL
        url = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_csv_2024-04-18.zip"
        
        logger.info("Downloading USDA FoodData Central dataset...")
        response = requests.get(url, stream=True)
        zip_path = os.path.join(output_dir, 'fooddata.zip')
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        logger.info("USDA data downloaded and extracted successfully")
        return output_dir
    
    def load_products_from_usda(self, usda_dir: str = None):
        """Load and process USDA product data into SQL Server - now uses real data already loaded"""
        
        # Check if we already have real USDA data loaded
        try:
            products_query = "SELECT COUNT(*) as count FROM Products WHERE ProductCode LIKE 'USDA_%'"
            result = pd.read_sql(products_query, self.engine)
            real_products_count = result['count'].iloc[0]
            
            if real_products_count > 0:
                logger.info(f"Using existing real USDA data: {real_products_count:,} products")
                
                # Get sample of real products for processing
                sample_query = """
                SELECT TOP 1000 
                    ProductCode, ProductName, Category, Subcategory, 
                    ShelfLifeDays, OptimalTempMin, OptimalTempMax,
                    OptimalHumidityMin, OptimalHumidityMax, UnitCost, UnitPrice
                FROM Products 
                WHERE ProductCode LIKE 'USDA_%'
                ORDER BY NEWID()
                """
                products_df = pd.read_sql(sample_query, self.engine)
                logger.info(f"Loaded {len(products_df)} real USDA products for processing")
                return products_df
        except Exception as e:
            logger.warning(f"Could not check existing data: {e}")
        
        # Fallback to loading from file if needed
        if usda_dir and os.path.exists(usda_dir):
            food_df = pd.read_csv(os.path.join(usda_dir, 'food.csv'))
            
            # Filter for fresh produce using keywords (same as our real data loader)
            fresh_keywords = [
                'apple', 'banana', 'orange', 'strawberry', 'grape', 'watermelon',
                'pineapple', 'mango', 'peach', 'pear', 'plum', 'cherry',
                'tomato', 'cucumber', 'lettuce', 'carrot', 'potato', 'onion',
                'broccoli', 'spinach', 'cabbage', 'pepper', 'milk', 'cheese',
                'yogurt', 'butter', 'egg', 'fresh', 'organic'
            ]
            
            fresh_foods = food_df[
                food_df['description'].str.lower().str.contains('|'.join(fresh_keywords), na=False)
            ]
            
            # Create products dataframe with real data processing
            products_data = []
            
            for _, row in fresh_foods.head(1000).iterrows():  # Limit for processing
                description = row['description'].lower()
                
                # Categorize based on description (same logic as real data loader)
                if any(fruit in description for fruit in ['apple', 'banana', 'orange', 'strawberry', 'grape', 'watermelon', 'pineapple', 'mango', 'peach', 'pear', 'plum', 'cherry']):
                    category = 'Fruits'
                elif any(veg in description for veg in ['tomato', 'cucumber', 'lettuce', 'carrot', 'potato', 'onion', 'broccoli', 'spinach', 'cabbage', 'pepper']):
                    category = 'Vegetables'
                elif any(dairy in description for dairy in ['milk', 'cheese', 'yogurt', 'butter', 'cream']):
                    category = 'Dairy'
                else:
                    category = 'Other Fresh Produce'
                
                # Estimate shelf life based on category and description
                if 'leaf' in description or 'lettuce' in description:
                    shelf_days = np.random.randint(3, 8)
                elif 'root' in description or 'potato' in description:
                    shelf_days = np.random.randint(14, 30)
                elif 'dairy' in description or 'milk' in description:
                    shelf_days = np.random.randint(7, 30)
                elif any(fruit in description for fruit in ['apple', 'banana', 'orange', 'strawberry', 'grape']):
                    shelf_days = np.random.randint(3, 21)
                else:
                    shelf_days = np.random.randint(3, 14)
                
                # Temperature ranges based on real storage requirements
                if 'dairy' in description or 'milk' in description:
                    temp_min, temp_max = 1, 4
                elif 'leaf' in description or 'lettuce' in description:
                    temp_min, temp_max = 0, 2
                else:
                    temp_min, temp_max = 2, 6
                
                product = {
                    'ProductCode': f"USDA_{row['fdc_id']}",
                    'ProductName': row['description'][:200],
                    'Category': category,
                    'Subcategory': row.get('data_type', 'Fresh Produce'),
                    'ShelfLifeDays': shelf_days,
                    'OptimalTempMin': temp_min,
                    'OptimalTempMax': temp_max,
                    'OptimalHumidityMin': 85.0,
                    'OptimalHumidityMax': 95.0,
                    'UnitCost': np.random.uniform(0.5, 5.0),
                    'UnitPrice': np.random.uniform(1.0, 8.0)
                }
                products_data.append(product)
            
            products_df = pd.DataFrame(products_data)
            logger.info(f"Processed {len(products_df)} products from USDA data")
            return products_df
        
        # If no data available, return empty dataframe
        logger.warning("No USDA data available")
        return pd.DataFrame()
    
    def generate_synthetic_iot_data(self, days: int = 30):
        """Generate realistic IoT sensor data using real warehouse locations and Norwegian climate"""
        
        # Get real warehouses from database
        warehouses_query = """
        SELECT WarehouseID, WarehouseCode, WarehouseName, LocationLat, LocationLon, Country, Region
        FROM Warehouses
        """
        warehouses = pd.read_sql(warehouses_query, self.engine)
        
        if warehouses.empty:
            logger.warning("No warehouses found, creating sample data")
            self._create_sample_warehouses()
            warehouses = pd.read_sql(warehouses_query, self.engine)
        
        # Check if we already have IoT data
        try:
            iot_count_query = "SELECT COUNT(*) as count FROM TemperatureLogs"
            result = pd.read_sql(iot_count_query, self.engine)
            existing_count = result['count'].iloc[0]
            
            if existing_count > 1000:  # If we have substantial data, use it
                logger.info(f"Using existing IoT data: {existing_count:,} readings")
                return
        except Exception as e:
            logger.warning(f"Could not check existing IoT data: {e}")
        
        # Generate realistic temperature logs based on Norwegian climate
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        iot_data = []
        
        for warehouse in warehouses.itertuples():
            # Realistic Norwegian climate data
            base_temp = 4.0  # Typical cold storage temperature
            if 'Oslo' in warehouse.WarehouseName:
                base_temp = 3.5
            elif 'Bergen' in warehouse.WarehouseName:
                base_temp = 4.5  # Slightly warmer due to Gulf Stream
            elif 'Trondheim' in warehouse.WarehouseName:
                base_temp = 3.0  # Colder northern climate
            elif 'Stockholm' in warehouse.WarehouseName:
                base_temp = 3.8  # Swedish climate
            elif 'Copenhagen' in warehouse.WarehouseName:
                base_temp = 4.2  # Danish climate
            
            # 5 devices per warehouse
            for device_num in range(1, 6):
                device_id = f"SENSOR_{warehouse.WarehouseCode}_{device_num:02d}"
                
                # Generate readings every 30 minutes (more realistic)
                current_time = start_date
                
                while current_time < end_date:
                    # Realistic temperature variations
                    temp_variation = np.random.normal(0, 0.3)
                    
                    # Occasional temperature spikes (equipment issues)
                    if np.random.random() < 0.02:  # 2% chance
                        temp_variation += np.random.uniform(1, 3)
                    
                    temperature = base_temp + temp_variation
                    humidity = np.random.uniform(88, 95)  # High humidity for fresh produce
                    co2 = np.random.uniform(400, 500)  # Normal CO2 levels
                    ethylene = np.random.uniform(0.01, 0.05)  # Low ethylene for fresh produce
                    
                    # Quality score based on real conditions
                    if temperature < 0 or temperature > 8:
                        quality_score = 0.3  # Poor quality
                    elif 2 <= temperature <= 6:
                        quality_score = 0.9  # Excellent quality
                    else:
                        quality_score = 0.7  # Good quality
                    
                    iot_data.append({
                        'LogTime': current_time,
                        'DeviceID': device_id,
                        'WarehouseID': warehouse.WarehouseID,
                        'Zone': f"Zone_{np.random.randint(1, 4)}",
                        'Temperature': round(temperature, 2),
                        'Humidity': round(humidity, 2),
                        'CO2Level': round(co2, 2),
                        'EthyleneLevel': round(ethylene, 4),
                        'QualityScore': round(quality_score, 2)
                    })
                    
                    current_time += timedelta(minutes=30)
                    
                    if len(iot_data) >= 5000:  # Batch insert
                        df = pd.DataFrame(iot_data)
                        df.to_sql('TemperatureLogs', self.engine, if_exists='append', index=False)
                        logger.info(f"Inserted {len(iot_data)} temperature logs")
                        iot_data = []
        
        # Insert remaining data
        if iot_data:
            df = pd.DataFrame(iot_data)
            df.to_sql('TemperatureLogs', self.engine, if_exists='append', index=False)
            logger.info(f"Inserted final {len(iot_data)} temperature logs")
    
    def _create_sample_warehouses(self):
        """Create sample warehouses for Norway and Nordic region"""
        
        warehouses = [
            {'WarehouseCode': 'OSL_01', 'WarehouseName': 'Oslo Central Distribution', 
             'LocationLat': 59.9139, 'LocationLon': 10.7522, 'Country': 'Norway', 'Region': 'Eastern'},
            {'WarehouseCode': 'BGO_01', 'WarehouseName': 'Bergen Fresh Hub', 
             'LocationLat': 60.3913, 'LocationLon': 5.3221, 'Country': 'Norway', 'Region': 'Western'},
            {'WarehouseCode': 'TRD_01', 'WarehouseName': 'Trondheim Cold Storage', 
             'LocationLat': 63.4305, 'LocationLon': 10.3951, 'Country': 'Norway', 'Region': 'Central'},
            {'WarehouseCode': 'STO_01', 'WarehouseName': 'Stockholm Import Center', 
             'LocationLat': 59.3293, 'LocationLon': 18.0686, 'Country': 'Sweden', 'Region': 'Nordic'},
            {'WarehouseCode': 'CPH_01', 'WarehouseName': 'Copenhagen Distribution', 
             'LocationLat': 55.6761, 'LocationLon': 12.5683, 'Country': 'Denmark', 'Region': 'Nordic'}
        ]
        
        for wh in warehouses:
            wh['CapacityUnits'] = np.random.randint(10000, 50000)
            wh['TemperatureControlled'] = 1
        
        df = pd.DataFrame(warehouses)
        df.to_sql('Warehouses', self.engine, if_exists='append', index=False)
        logger.info(f"Created {len(warehouses)} sample warehouses")
    
    def load_fruits_360_dataset(self, data_dir: str = './data/fruits360'):
        """Download and process Fruits-360 dataset for computer vision"""
        
        # This would download from Kaggle API if configured
        # For now, we'll create a reference structure
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Create dataset structure reference
        dataset_info = {
            'total_images': 90483,
            'classes': 131,
            'training_set': 67692,
            'test_set': 22688,
            'image_size': (100, 100),
            'categories': [
                'Apple', 'Banana', 'Orange', 'Strawberry', 'Grape',
                'Watermelon', 'Pineapple', 'Mango', 'Papaya', 'Kiwi',
                'Peach', 'Pear', 'Plum', 'Cherry', 'Apricot',
                'Avocado', 'Lemon', 'Lime', 'Tomato', 'Cucumber'
            ]
        }
        
        # Save dataset info
        with open(os.path.join(data_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Fruits-360 dataset reference created at {data_dir}")
        return dataset_info
    
    def create_sample_inventory(self, num_products: int = 100, num_lots: int = 500):
        """Create sample inventory data for testing"""
        
        # Get products and warehouses
        products = pd.read_sql("SELECT ProductID, ProductCode, ShelfLifeDays FROM Products", self.engine)
        warehouses = pd.read_sql("SELECT WarehouseID, WarehouseCode FROM Warehouses", self.engine)
        
        if products.empty or warehouses.empty:
            logger.error("No products or warehouses found. Please load data first.")
            return
        
        inventory_data = []
        
        for _ in range(num_lots):
            product = products.sample(1).iloc[0]
            warehouse = warehouses.sample(1).iloc[0]
            
            # Generate lot data
            production_date = datetime.now() - timedelta(days=np.random.randint(0, 30))
            expiry_date = production_date + timedelta(days=product['ShelfLifeDays'])
            
            # Skip if already expired
            if expiry_date < datetime.now():
                continue
            
            lot_data = {
                'ProductID': product['ProductID'],
                'WarehouseID': warehouse['WarehouseID'],
                'LotNumber': f"LOT_{product['ProductCode']}_{np.random.randint(1000, 9999)}",
                'Quantity': np.random.randint(10, 1000),
                'ProductionDate': production_date.date(),
                'ExpiryDate': expiry_date.date(),
                'Status': 'FRESH' if expiry_date > datetime.now() + timedelta(days=2) else 'WARNING'
            }
            
            inventory_data.append(lot_data)
        
        inventory_df = pd.DataFrame(inventory_data)
        inventory_df.to_sql('Inventory', self.engine, if_exists='append', index=False)
        logger.info(f"Created {len(inventory_df)} inventory lots")
        
        return inventory_df
    
    def create_sample_waste_events(self, num_events: int = 200):
        """Create sample waste events for analysis"""
        
        # Get inventory data
        inventory = pd.read_sql("""
            SELECT i.*, p.ProductName, p.Category 
            FROM Inventory i 
            JOIN Products p ON i.ProductID = p.ProductID
            WHERE i.Quantity > 0
        """, self.engine)
        
        if inventory.empty:
            logger.error("No inventory found. Please create inventory first.")
            return
        
        waste_events = []
        waste_reasons = ['Expired', 'Temperature Violation', 'Quality Issue', 'Damaged', 'Overstock']
        
        for _ in range(num_events):
            lot = inventory.sample(1).iloc[0]
            
            # Determine waste reason based on expiry
            if lot['DaysUntilExpiry'] <= 0:
                reason = 'Expired'
            elif lot['DaysUntilExpiry'] <= 2:
                reason = np.random.choice(['Expired', 'Quality Issue'])
            else:
                reason = np.random.choice(waste_reasons, p=[0.1, 0.2, 0.3, 0.2, 0.2])
            
            waste_quantity = min(lot['Quantity'], np.random.randint(1, 50))
            
            waste_event = {
                'ProductID': lot['ProductID'],
                'WarehouseID': lot['WarehouseID'],
                'LotNumber': lot['LotNumber'],
                'QuantityWasted': waste_quantity,
                'WasteReason': reason,
                'WasteCategory': 'Fresh Produce',
                'EstimatedValueLoss': waste_quantity * np.random.uniform(1.0, 5.0),
                'TemperatureViolation': reason == 'Temperature Violation',
                'DaysPastOptimal': max(0, -lot['DaysUntilExpiry'])
            }
            
            waste_events.append(waste_event)
        
        waste_df = pd.DataFrame(waste_events)
        waste_df.to_sql('WasteEvents', self.engine, if_exists='append', index=False)
        logger.info(f"Created {len(waste_df)} waste events")
        
        return waste_df