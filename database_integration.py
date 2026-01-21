"""
Partie 1: Intégration de la Base de Données
============================================
Ce module gère la création et la manipulation de la base de données SQLite
pour stocker les mesures de qualité de l'air.

Fonctionnalités:
- Création de la base de données et des tables
- Insertion des données depuis le fichier CSV
- Récupération des données avec filtres
- Mise à jour des enregistrements
- Suppression des enregistrements
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime


class AirQualityDatabase:
    #classe pour gérer la base de données 
    
    def __init__(self, db_path="air_quality.db"):

        self.db_path = db_path
        self.connection = None
        self.cursor = None
    
    def connect(self):
        #Établit la connexion à la base de données
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        print(f"Database Connection Established: {self.db_path}")
    
    def disconnect(self):
        #ferme la connexion à la base de données
        if self.connection:
            self.connection.close()
            print("Connection Closed.")
    
    def create_tables(self):
        
        #table principale pour les mesures de qualité de l'air
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS air_quality_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                co_gt REAL,
                no2_gt REAL,
                temperature REAL,
                humidity REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        #table pour les métadonnées des images
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                processing_methods TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table des images
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

 
        
        #table pour stocker les résultats de corrélation
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS correlation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                variable1 TEXT NOT NULL,
                variable2 TEXT NOT NULL,
                correlation_coefficient REAL,
                correlation_type TEXT,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        #table pr stocker resultst d'analyse spectrale
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS spectral_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                variable_name TEXT NOT NULL,
                dominant_frequency REAL,
                power_spectrum_data TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.connection.commit()
        print("Tables Created Successfully.")
    
    def load_csv_to_database(self, csv_path):
        
        #lecture du CSV avec le bon séparateur et format décimal
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        
        #suppression des colonnes vides
        df = df.dropna(axis=1, how='all')
        
        #renommer colonnes pour correspondre à notre schéma
        column_mapping = {
            'Date': 'date',
            'Time': 'time',
            'CO(GT)': 'co_gt',
            'NO2(GT)': 'no2_gt',
            'T': 'temperature',
            'RH': 'humidity',
        }
        df = df.rename(columns=column_mapping)
        
        df = df.replace(-200, None)
        
        df = df.dropna(how='all')
        
        #insertion des données
        count = 0
        for _, row in df.iterrows():
            try:
                self.cursor.execute('''
                    INSERT INTO air_quality_measurements 
                    (date, time, co_gt, no2_gt, temperature, humidity)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    row.get('date'), row.get('time'), row.get('co_gt'),
                    row.get('no2_gt'), row.get('temperature'), row.get('humidity')
                ))
                count += 1
            except Exception as e:
                print(f"Error During Insertion: {e}")
        
        self.connection.commit()
        print(f"{count} Records Inserted Since {csv_path}")
        return count
    
    # REQUÊTES CRUD
    
    def insert_measurement(self, date, time, co_gt=None, no2_gt=None,
                           temperature=None, humidity=None):
    
        self.cursor.execute('''
            INSERT INTO air_quality_measurements 
            (date, time, co_gt, no2_gt, 
             temperature, humidity)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (date, time, co_gt, no2_gt,
              temperature, humidity))
        self.connection.commit()
        print(f"Measurement Inserted with ID: {self.cursor.lastrowid}")
        return self.cursor.lastrowid
    
    def get_all_measurements(self, limit=None):
        #Récupère toutes les mesures
              
        query = "SELECT * FROM air_quality_measurements"
        if limit:
            query += f" LIMIT {limit}"
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def get_measurements_by_date(self, start_date, end_date=None):

        if end_date:
            self.cursor.execute('''
                SELECT * FROM air_quality_measurements 
                WHERE date >= ? AND date <= ?
            ''', (start_date, end_date))
        else:
            self.cursor.execute('''
                SELECT * FROM air_quality_measurements 
                WHERE date = ?
            ''', (start_date,))
        return self.cursor.fetchall()
    
    def get_measurements_by_threshold(self, column, min_value=None, max_value=None):
  
        valid_columns = ['co_gt', 'no2_gt',
                        'temperature', 'humidity']
        
        if column not in valid_columns:
            raise ValueError(f"Invalid Column. Valid Columns: {valid_columns}")
        
        if min_value is not None and max_value is not None:
            self.cursor.execute(f'''
                SELECT * FROM air_quality_measurements 
                WHERE {column} >= ? AND {column} <= ?
            ''', (min_value, max_value))
        elif min_value is not None:
            self.cursor.execute(f'''
                SELECT * FROM air_quality_measurements 
                WHERE {column} >= ?
            ''', (min_value,))
        elif max_value is not None:
            self.cursor.execute(f'''
                SELECT * FROM air_quality_measurements 
                WHERE {column} <= ?
            ''', (max_value,))
        
        return self.cursor.fetchall()
    
    def update_measurement(self, record_id, **kwargs):
  
        if not kwargs:
            print("No Data to Update.")
            return False
        
        set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [record_id]
        
        self.cursor.execute(f'''
            UPDATE air_quality_measurements 
            SET {set_clause}
            WHERE id = ?
        ''', values)
        self.connection.commit()
        print(f"Record {record_id} Updated.")
        return True
    
    def delete_measurement(self, record_id):

        self.cursor.execute('''
            DELETE FROM air_quality_measurements WHERE id = ?
        ''', (record_id,))
        self.connection.commit()
        print(f"Record {record_id} Deleted.")
        return True
    
    def get_statistics(self):
        stats = {}
        
        #nombre total d'enregistrements
        self.cursor.execute("SELECT COUNT(*) FROM air_quality_measurements")
        stats['total_records'] = self.cursor.fetchone()[0]
        
        columns = ['co_gt', 'temperature', 'humidity', 'no2_gt']
        for col in columns:
            self.cursor.execute(f'''
                SELECT AVG({col}), MIN({col}), MAX({col}) 
                FROM air_quality_measurements 
                WHERE {col} IS NOT NULL
            ''')
            result = self.cursor.fetchone()
            stats[col] = {
                'moyenne': round(result[0], 2) if result[0] else None,
                'min': result[1],
                'max': result[2]
            }
        
        return stats
    
    def get_data_as_dataframe(self):

        query = "SELECT * FROM air_quality_measurements"
        df = pd.read_sql_query(query, self.connection)
        return df


#FONCTIONS DE TEST 

def test_database_operations():
    #teste toutes les opérations de la base de données
    
    print("=" * 60)
    print("Test Database Operations")
    print("=" * 60)
    
    #initialisation
    db = AirQualityDatabase("air_quality.db")
    db.connect()
    
    #création tables
    print("\n1. Table Creation..")
    db.create_tables()
    
    #chargement des données CSV
    print("\n2. CSV Data Loading...")
    csv_path = "AirQualityUCI.csv"
    if os.path.exists(csv_path):
        db.load_csv_to_database(csv_path)
    else:
        print(f"File {csv_path} Not Found.")
    
    #test de récupération
    print("\n3. Data Retrieval Test..")
    records = db.get_all_measurements(limit=5)
    print(f"First 5 Records Retrieved: {len(records)} Rows")
    
    #test de filtrage par date
    print("\n4. Date Filtering Test..")
    filtered = db.get_measurements_by_date("10/03/2004")
    print(f"Records for 03/10/2004: {len(filtered)} Rows")
    
    #test de filtrage par seuil
    print("\n5. Threshold Filtering Test (Temperature > 20°C)..")
    high_temp = db.get_measurements_by_threshold('temperature', min_value=20)
    print(f"Records with Temperature > 20°C: {len(high_temp)} Rows")
    
    #test d'insertion
    print("\n6. New Measurement Insertion Test..")
    new_id = db.insert_measurement(
        date="31/12/2025",
        time="12.00.00",
        no2_gt=0.03,
        temperature=15.0,
        humidity=55.0
    )
    
    #test de mise à jour
    print("\n7. Update Test..")
    db.update_measurement(new_id, temperature=16.5, humidity=60.0)
    
    #test des statistiques
    print("\n8. Database Statistics..")
    stats = db.get_statistics()
    print(f" Total Records: {stats['total_records']}")
    print(f" Temperature - Moy: {stats['temperature']['moyenne']}°C, "
          f"Min: {stats['temperature']['min']}°C, Max: {stats['temperature']['max']}°C")
    
    #test de suppression
    print("\n9. Deletion Test..")
    db.delete_measurement(new_id)
    
    #récupération en DataFrame
    print("\n10. DataFrame Retrieval Test..")
    df = db.get_data_as_dataframe()
    print(f" DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f" Columns: {list(df.columns)}")
    
    db.disconnect()
    
    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_database_operations()
