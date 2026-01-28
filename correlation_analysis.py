import pandas as pd
from database_integration import AirQualityDatabase


class CorrelationAnalyzer:

    def __init__(self, db_path="db_air_quality"):
        self.db = AirQualityDatabase(db_path)
        self.data = None
        self.correlation_matrix = None
        self.numeric_columns = ['co_gt', 'no2_gt','temperature', 'humidity']

    def load_data(self):
        self.db.connect()
        self.data = self.db.get_data_as_dataframe()
        self.db.disconnect()
        
        available_cols = [col for col in self.numeric_columns if col in self.data.columns]
        self.data = self.data[available_cols].dropna()
        
        print(f"Data Loaded: {len(self.data)} Records")
        print(f"Variables: {available_cols}")
        return self.data
    
    def calculate_pearson_correlation(self):
        if self.data is None:
            self.load_data()
        
        self.correlation_matrix = self.data.corr(method='pearson')
        print("Pearson Correlation Calculated")
        return self.correlation_matrix
    
    def calculate_spearman_correlation(self):
        if self.data is None:
            self.load_data()
        
        self.correlation_matrix = self.data.corr(method='spearman')
        print("Spearman Correlation Calculated")
        return self.correlation_matrix
    
    def store_correlation_results(self, method='pearson'):
        if method == 'pearson':
            corr_matrix = self.calculate_pearson_correlation()
        else:
            corr_matrix = self.calculate_spearman_correlation()
        
        self.db.connect()
        
        self.db.cursor.execute("DELETE FROM correlation_results WHERE correlation_type = %s", (method,))
        
        count = 0
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                coef = corr_matrix.iloc[i, j]
                
                self.db.cursor.execute('''
                    INSERT INTO correlation_results 
                    (variable1, variable2, correlation_coefficient, correlation_type)
                    VALUES (%s, %s, %s, %s)
                ''', (var1, var2, float(coef), method))
                count += 1
        
        self.db.connection.commit()
        self.db.disconnect()
        
        print(f"{count} Correlation Results Stored in Database (méthode: {method})")

