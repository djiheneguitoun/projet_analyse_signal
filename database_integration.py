import mysql.connector
import pandas as pd


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

