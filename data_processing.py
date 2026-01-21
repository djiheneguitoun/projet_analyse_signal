"""
Partie 2: Chargement et Filtrage des Données
=============================================
Ce module gère le chargement, nettoyage et filtrage des données
de qualité de l'air avec des techniques de réduction de bruit.

Fonctionnalités:
- Importation et nettoyage des données CSV
- Stockage des données nettoyées dans la base
- Filtrage par moyenne mobile
- Filtrage par seuil
- Détection et suppression des valeurs aberrantes
"""

import pandas as pd
import numpy as np
from database_integration import AirQualityDatabase
import matplotlib.pyplot as plt


class DataProcessor:
    #classe pour le chargement et filtrage des données
    
    def __init__(self, db_path="air_quality.db"):

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
    
    def load_data_from_database(self):

        self.db.connect()
        self.data = self.db.get_data_as_dataframe()
        self.db.disconnect()
        print(f" Data Loaded from Database: {len(self.data)} Records")
        return self.data
    
    def clean_data(self):

        if self.data is None:
            raise ValueError("No Data Loaded. Use load_data_from_csv() or load_data_from_database().")
        
        self.cleaned_data = self.data.copy()
        
        numeric_cols = ['co_gt','no2_gt',
                       'temperature', 'humidity', ]
        
        for col in numeric_cols:
            if col in self.cleaned_data.columns:
                self.cleaned_data[col] = self.cleaned_data[col].replace(-200, np.nan)
        
        #compter les valeurs manquantes avant nettoyage
        missing_before = self.cleaned_data[numeric_cols].isnull().sum().sum()
        
        #suppression des lignes avec trop de valeurs manquantes (>50%)
        threshold = len(numeric_cols) * 0.5
        self.cleaned_data = self.cleaned_data.dropna(thresh=len(self.cleaned_data.columns) - threshold)
        
        #interpolation linéaire pour les valeurs manquantes restantes
        for col in numeric_cols:
            if col in self.cleaned_data.columns:
                self.cleaned_data[col] = self.cleaned_data[col].interpolate(method='linear')
        
        #remplir les valeurs restantes avec la médiane
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
    
    def apply_moving_average(self, column, window_size=5):
     
        if self.cleaned_data is None:
            self.clean_data()
        
        if column not in self.cleaned_data.columns:
            raise ValueError(f"Column '{column}' Not Found.")
        
        filtered = self.cleaned_data[column].rolling(window=window_size, center=True).mean()
        
        #remplir les valeurs NaN aux extrémités
        filtered = filtered.fillna(method='bfill').fillna(method='ffill')
        
        print(f"Moving Average Applied on '{column}' (fenêtre={window_size})")
        return filtered
    
    def apply_threshold_filter(self, column, min_value=None, max_value=None):
        
        if self.cleaned_data is None:
            self.clean_data()
        
        if column not in self.cleaned_data.columns:
            raise ValueError(f"Column '{column}' Not Found.")
        
        filtered_data = self.cleaned_data.copy()
        
        if min_value is not None:
            filtered_data = filtered_data[filtered_data[column] >= min_value]
        
        if max_value is not None:
            filtered_data = filtered_data[filtered_data[column] <= max_value]
        filtered_data = filtered_data.reset_index(drop=True)
        
        print(f" Threshold Filter Applied on '{column}':")
        print(f"  - Min: {min_value}, Max: {max_value}")
        print(f"  - Records After Filtering : {len(filtered_data)}")
        
        return filtered_data
    
    def remove_outliers(self, column, method='iqr', threshold=1.5):

        if self.cleaned_data is None:
            self.clean_data()
        
        data = self.cleaned_data.copy()
        original_len = len(data)
        
        if method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
            
        elif method == 'zscore':
            mean = data[column].mean()
            std = data[column].std()
            data = data[abs((data[column] - mean) / std) <= threshold]
        
        removed = original_len - len(data)
        print(f" Outliers Removed from'{column}' (méthode={method}):")
        print(f"  - Values Removed: {removed}")
        print(f"  - Remaining Records: {len(data)}")
        
        return data
    
    def get_summary_statistics(self):
     
        if self.cleaned_data is None:
            self.clean_data()
        
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        stats = self.cleaned_data[numeric_cols].describe()
        
        print("Descriptive Statistics Calculated")
        return stats
    
    def store_cleaned_data(self):

        if self.cleaned_data is None:
            self.clean_data()
        
        self.db.connect()
        
        self.db.create_tables()
        
        self.db.cursor.execute("DELETE FROM air_quality_measurements")
        
        #insérer les données nettoyées
        count = 0
        for _, row in self.cleaned_data.iterrows():
            try:
                self.db.cursor.execute('''
                    INSERT INTO air_quality_measurements 
                    (date, time, co_gt, no2_gt, 
                     temperature, humidity)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    row.get('date'), row.get('time'), row.get('co_gt'),
                    row.get('no2_gt'), row.get('temperature'), row.get('humidity')
                ))
                count += 1
            except Exception as e:
                print(f"Error: {e}")
        
        self.db.connection.commit()
        self.db.disconnect()
        
        print(f"{count} Cleaned Records Stored in the Database")
    
    def visualize_filtering_effect(self, column, window_size=10, save_path=None):
        
        if self.cleaned_data is None:
            self.clean_data()
        
        original = self.cleaned_data[column].values[:500]  #limiter pour la visualisation
        filtered = self.apply_moving_average(column, window_size).values[:500]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        #données originales
        axes[0].plot(original, 'b-', alpha=0.7, linewidth=0.8)
        axes[0].set_title(f'{column} - Original Data')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Moyenne mobile
        axes[1].plot(original, 'b-', alpha=0.3, linewidth=0.5, label='Original')
        axes[1].plot(filtered, 'r-', linewidth=1.5, label=f'Moving Average (n={window_size})')
        axes[1].set_title(f'{column} - Moving Average Filter')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure Saved: {save_path}")
        
        plt.show()


def test_data_processing():
    
    print("=" * 60)
    print("DATA PROCESSING TEST")
    print("=" * 60)
    
    processor = DataProcessor()
    
    print("\n1. Loading Data from CSV..")
    processor.load_data_from_csv("AirQualityUCI.csv")
    
    print("\n2. Data Cleaning..")
    processor.clean_data()
    
    print("\n3. Descriptive Statistics:")
    stats = processor.get_summary_statistics()
    print(stats[['temperature', 'humidity', 'co_gt']].round(2))
    
    print("\n4. Applying Moving Average on 'Temperature'..")
    temp_filtered = processor.apply_moving_average('temperature', window_size=10)
    print(f" Preview: {temp_filtered[:5].values}")
    
    print("\n5. Threshold Filtering (Temperature Between 10°C and 30°C)..")
    filtered_data = processor.apply_threshold_filter('temperature', min_value=10, max_value=30)
    
    print("\n6. Outlier Removal (IQR Method)..")
    no_outliers = processor.remove_outliers('co_gt', method='iqr', threshold=1.5)
    

    print("\n8. Storing Cleaned Data in the Database..")
    processor.store_cleaned_data()
    
    print("\n9. Creating the Visualization..")
    processor.visualize_filtering_effect('temperature', window_size=10, 
                                         save_path='images/filtering_effect.png')
    
    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_processing()
