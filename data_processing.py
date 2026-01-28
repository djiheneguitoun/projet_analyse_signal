import pandas as pd
import numpy as np
from database_integration import AirQualityDatabase


class DataProcessor:
    
    def __init__(self, db_path="db_air_quality"):
        self.db = AirQualityDatabase(db_path)
        self.data = None
        self.cleaned_data = None
    
    def load_data_from_csv(self, csv_path):
        self.data = pd.read_csv(csv_path, sep=';', decimal=',')
        self.data = self.data.dropna(axis=1, how='all')
        
        column_mapping = {
            'Date': 'date',
            'Time': 'time',
            'CO(GT)': 'co_gt',
            'NO2(GT)': 'no2_gt',
            'T': 'temperature',
            'RH': 'humidity',
        }
        self.data = self.data.rename(columns=column_mapping)
        
        print(f" Data Loaded: {len(self.data)} Records")
        print(f" Columns: {list(self.data.columns)}")
        return self.data
    
    def clean_data(self):
        if self.data is None:
            raise ValueError("No Data Loaded. Use load_data_from_csv() or load_data_from_database().")
        
        self.cleaned_data = self.data.copy()
        
        numeric_cols = ['co_gt','no2_gt', 'temperature', 'humidity']
        
        for col in numeric_cols:
            if col in self.cleaned_data.columns:
                self.cleaned_data[col] = self.cleaned_data[col].replace(-200, np.nan)
        
        missing_before = self.cleaned_data[numeric_cols].isnull().sum().sum()
        
        threshold = len(numeric_cols) * 0.5
        self.cleaned_data = self.cleaned_data.dropna(thresh=len(self.cleaned_data.columns) - threshold)
        
        for col in numeric_cols:
            if col in self.cleaned_data.columns:
                self.cleaned_data[col] = self.cleaned_data[col].interpolate(method='linear')
        
        for col in numeric_cols:
            if col in self.cleaned_data.columns:
                median_val = self.cleaned_data[col].median()
                self.cleaned_data[col] = self.cleaned_data[col].fillna(median_val)
        
        missing_after = self.cleaned_data[numeric_cols].isnull().sum().sum()
        
        print(f" Data Cleaned:")
        print(f"  - Missing Values Before: {missing_before}")
        print(f"  - Missing Values After: {missing_after}")
        print(f"  - Remaining Records: {len(self.cleaned_data)}")
        
        return self.cleaned_data
    
    def store_cleaned_data(self):
        if self.cleaned_data is None:
            self.clean_data()
        
        self.db.connect()
        self.db.create_tables()
        self.db.cursor.execute("DELETE FROM air_quality_measurements")
        
        count = 0
        for _, row in self.cleaned_data.iterrows():
            try:
                self.db.cursor.execute('''
                    INSERT INTO air_quality_measurements 
                    (date, time, co_gt, no2_gt, 
                     temperature, humidity)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (
                    row.get('date'), row.get('time'), 
                    None if pd.isna(row.get('co_gt')) else row.get('co_gt'),
                    None if pd.isna(row.get('no2_gt')) else row.get('no2_gt'),
                    None if pd.isna(row.get('temperature')) else row.get('temperature'),
                    None if pd.isna(row.get('humidity')) else row.get('humidity')
                ))
                count += 1
            except Exception as e:
                print(f"Error: {e}")
        
        self.db.connection.commit()
        self.db.disconnect()
        
        print(f"{count} Cleaned Records Stored in the Database")

