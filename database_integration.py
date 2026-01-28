import mysql.connector
import pandas as pd
import os
from datetime import datetime


class AirQualityDatabase:
    
    def __init__(self, db_path="db_air_quality"):
        self.db_name = db_path if db_path != "air_quality.db" else "db_air_quality"
        self.host = "localhost"
        self.user = "root"
        self.password = ""
        self.connection = None
        self.cursor = None
    
    def connect(self):
        try:
            temp_conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            temp_cursor = temp_conn.cursor()
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")
            temp_conn.close()
            
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.db_name
            )
            self.cursor = self.connection.cursor()
            print(f"Database Connection Established: {self.db_name}")
        except mysql.connector.Error as err:
            print(f"MySQL Error: {err}")
            raise
    
    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("Connection Closed.")
    
    def create_tables(self):
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS air_quality_measurements (
                id INT AUTO_INCREMENT PRIMARY KEY,
                date VARCHAR(20) NOT NULL,
                time VARCHAR(20) NOT NULL,
                co_gt FLOAT,
                no2_gt FLOAT,
                temperature FLOAT,
                humidity FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_metadata (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                file_path VARCHAR(500),
                file_size INT,
                width INT,
                height INT,
                processing_methods TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        ''')

 
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS correlation_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                variable1 VARCHAR(100) NOT NULL,
                variable2 VARCHAR(100) NOT NULL,
                correlation_coefficient FLOAT,
                correlation_type VARCHAR(50),
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS spectral_analysis (
                id INT AUTO_INCREMENT PRIMARY KEY,
                variable_name VARCHAR(100) NOT NULL,
                dominant_frequency FLOAT,
                power_spectrum_data TEXT,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS filtered_data_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                original_record_id INT,
                variable_name VARCHAR(100) NOT NULL,
                filter_type VARCHAR(100) NOT NULL,
                window_size INT,
                threshold_min FLOAT,
                threshold_max FLOAT,
                original_value FLOAT,
                filtered_value FLOAT,
                row_index INT,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (original_record_id) REFERENCES air_quality_measurements(id) ON DELETE CASCADE
            )
        ''')
        
        self.connection.commit()
        print("Tables Created Successfully.")
    
    def load_csv_to_database(self, csv_path):
        
        #lecture du CSV avec le bon séparateur et format décimal
        df = pd.read_csv(csv_path, sep=';', decimal=',')
        
        df = df.dropna(axis=1, how='all')
        
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

        count = 0
        for _, row in df.iterrows():
            try:
                self.cursor.execute('''
                    INSERT INTO air_quality_measurements 
                    (date, time, co_gt, no2_gt, temperature, humidity)
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
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (date, time, co_gt, no2_gt,
              temperature, humidity))
        self.connection.commit()
        print(f"Measurement Inserted with ID: {self.cursor.lastrowid}")
        return self.cursor.lastrowid
    
    def get_all_measurements(self, limit=None):
              
        query = "SELECT * FROM air_quality_measurements"
        if limit:
            query += f" LIMIT {limit}"
        self.cursor.execute(query)
        return self.cursor.fetchall()
    
    def get_measurements_by_date(self, start_date, end_date=None):

        if end_date:
            self.cursor.execute('''
                SELECT * FROM air_quality_measurements 
                WHERE date >= %s AND date <= %s
            ''', (start_date, end_date))
        else:
            self.cursor.execute('''
                SELECT * FROM air_quality_measurements 
                WHERE date = %s
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
                WHERE {column} >= %s AND {column} <= %s
            ''', (min_value, max_value))
        elif min_value is not None:
            self.cursor.execute(f'''
                SELECT * FROM air_quality_measurements 
                WHERE {column} >= %s
            ''', (min_value,))
        elif max_value is not None:
            self.cursor.execute(f'''
                SELECT * FROM air_quality_measurements 
                WHERE {column} <= %s
            ''', (max_value,))
        
        return self.cursor.fetchall()
    
    def update_measurement(self, record_id, **kwargs):
  
        if not kwargs:
            print("No Data to Update.")
            return False
        
        set_clause = ", ".join([f"{k} = %s" for k in kwargs.keys()])
        values = list(kwargs.values()) + [record_id]
        
        self.cursor.execute(f'''
            UPDATE air_quality_measurements 
            SET {set_clause}
            WHERE id = %s
        ''', values)
        self.connection.commit()
        print(f"Record {record_id} Updated.")
        return True
    
    def delete_measurement(self, record_id):

        self.cursor.execute('''
            DELETE FROM air_quality_measurements WHERE id = %s
        ''', (record_id,))
        self.connection.commit()
        print(f"Record {record_id} Deleted.")
        return True
    
    def insert_filtered_data(self, original_record_id, variable_name, filter_type, 
                            window_size, threshold_min, threshold_max, 
                            original_value, filtered_value, row_index):
        try:
            self.cursor.execute('''
                INSERT INTO filtered_data_history 
                (original_record_id, variable_name, filter_type, window_size, 
                 threshold_min, threshold_max, original_value, filtered_value, row_index)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (original_record_id, variable_name, filter_type, window_size,
                  threshold_min, threshold_max, original_value, filtered_value, row_index))
            self.connection.commit()
            return self.cursor.lastrowid
        except mysql.connector.Error as err:
            print(f"Error inserting filtered data: {err}")
            raise
    
    def get_filtered_data_history(self, variable_name=None, filter_type=None, limit=None):
        query = "SELECT * FROM filtered_data_history WHERE 1=1"
        params = []
        
        if variable_name:
            query += " AND variable_name = %s"
            params.append(variable_name)
        
        if filter_type:
            query += " AND filter_type = %s"
            params.append(filter_type)
        
        query += " ORDER BY applied_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        self.cursor.execute(query, params if params else None)
        return self.cursor.fetchall()
    
    def get_statistics(self):
        stats = {}
        
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
    
    print("=" * 60)
    print("Test Database Operations")
    print("=" * 60)
    
    #initialisation
    db = AirQualityDatabase("db_air_quality")
    db.connect()
    
    print("\n1. Table Creation..")
    db.create_tables()

    print("\n2. CSV Data Loading...")
    csv_path = "AirQualityUCI.csv"
    if os.path.exists(csv_path):
        db.load_csv_to_database(csv_path)
    else:
        print(f"File {csv_path} Not Found.")
    
    print("\n3. Data Retrieval Test..")
    records = db.get_all_measurements(limit=5)
    print(f"First 5 Records Retrieved: {len(records)} Rows")
    
    print("\n4. Date Filtering Test..")
    filtered = db.get_measurements_by_date("10/03/2004")
    print(f"Records for 03/10/2004: {len(filtered)} Rows")
    
    print("\n5. Threshold Filtering Test (Temperature > 20°C)..")
    high_temp = db.get_measurements_by_threshold('temperature', min_value=20)
    print(f"Records with Temperature > 20°C: {len(high_temp)} Rows")
    
    print("\n6. New Measurement Insertion Test..")
    new_id = db.insert_measurement(
        date="31/12/2025",
        time="12.00.00",
        no2_gt=0.03,
        temperature=15.0,
        humidity=55.0
    )

    print("\n7. Update Test..")
    db.update_measurement(new_id, temperature=16.5, humidity=60.0)
    
    print("\n8. Database Statistics..")
    stats = db.get_statistics()
    print(f" Total Records: {stats['total_records']}")
    print(f" Temperature - Moy: {stats['temperature']['moyenne']}°C, "
          f"Min: {stats['temperature']['min']}°C, Max: {stats['temperature']['max']}°C")
    
    print("\n9. Deletion Test..")
    db.delete_measurement(new_id)
    
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
