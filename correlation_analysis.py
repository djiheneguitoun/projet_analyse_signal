"""
Partie 3: Analyse des Corrélations
===================================
Ce module calcule et visualise les corrélations entre les variables
de qualité de l'air.

Fonctionnalités:
- Calcul des coefficients de corrélation (Pearson et Spearman)
- Stockage des résultats dans la base de données
- Visualisation avec scatter plots et heatmaps
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from database_integration import AirQualityDatabase
from data_processing import DataProcessor

LABELS = {
    'co_gt': 'CO',
    'no2_gt': 'NO2',
    'temperature': 'Temperature',
    'humidity': 'Humidity'
}

class CorrelationAnalyzer:

    def __init__(self, db_path="air_quality.db"):
        

        self.db = AirQualityDatabase(db_path)
        self.data = None
        self.correlation_matrix = None
        self.numeric_columns = ['co_gt', 'no2_gt','temperature', 
                                'humidity']
    def get_label(self, col):
        """Retourne le nom lisible d'une colonne."""
        return LABELS.get(col, col)

    
    def load_data(self):
        self.db.connect()
        self.data = self.db.get_data_as_dataframe()
        self.db.disconnect()
        
        #garder uniquement les colonnes numériques
        available_cols = [col for col in self.numeric_columns if col in self.data.columns]
        self.data = self.data[available_cols].dropna()
        
        print(f"Data Loaded: {len(self.data)} Records")
        print(f"Variables: {available_cols}")
        return self.data
    
    def calculate_pearson_correlation(self):
        #Calcule la matrice de corrélation de Pearson.
    
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
    
    def calculate_correlation_pair(self, var1, var2, method='pearson'):
        
        #Calcule la corrélation entre deux variables spécifiques.
       
        if self.data is None:
            self.load_data()
        
        if var1 not in self.data.columns or var2 not in self.data.columns:
            raise ValueError(f"Variables Not Found. Available: {list(self.data.columns)}")
        
        x = self.data[var1].values
        y = self.data[var2].values
        
        if method == 'pearson':
            coef, pvalue = stats.pearsonr(x, y)
        else:
            coef, pvalue = stats.spearmanr(x, y)
        
        print(f"Correlation {method} between '{var1}' et '{var2}':")
        print(f"  Coefficient: {coef:.4f}")
        print(f"  P-value: {pvalue:.2e}")
        
        return coef, pvalue
    
    def get_strongest_correlations(self, n=10, method='pearson'):
       #Trouve les N corrélations les plus fortes.
    
        if method == 'pearson':
            corr_matrix = self.calculate_pearson_correlation()
        else:
            corr_matrix = self.calculate_spearman_correlation()
        
        # Extraire les paires uniques
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                coef = corr_matrix.iloc[i, j]
                correlations.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Coefficient': coef,
                    'Abs Coefficient': abs(coef)
                })
        
        df = pd.DataFrame(correlations)
        df = df.sort_values('Abs Coefficient', ascending=False).head(n)
        df = df.drop('Abs Coefficient', axis=1)
        
        print(f"\n Top {n} correlations ({method}):")
        print(df.to_string(index=False))
        
        return df
    
    def store_correlation_results(self, method='pearson'):
        
        if method == 'pearson':
            corr_matrix = self.calculate_pearson_correlation()
        else:
            corr_matrix = self.calculate_spearman_correlation()
        
        self.db.connect()
        
        #vider les anciens résults
        self.db.cursor.execute("DELETE FROM correlation_results WHERE correlation_type = ?", (method,))
        
        #et insérer les nouvelles corrélations
        count = 0
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                coef = corr_matrix.iloc[i, j]
                
                self.db.cursor.execute('''
                    INSERT INTO correlation_results 
                    (variable1, variable2, correlation_coefficient, correlation_type)
                    VALUES (?, ?, ?, ?)
                ''', (var1, var2, float(coef), method))
                count += 1
        
        self.db.connection.commit()
        self.db.disconnect()
        
        print(f"{count} Correlation Results Stored in Database (méthode: {method})")
    
    def plot_heatmap(self, method='pearson', save_path=None):
        #crée une heatmap des corrélations
    
        if method == 'pearson':
            corr_matrix = self.calculate_pearson_correlation()
        else:
            corr_matrix = self.calculate_spearman_correlation()
        
        plt.figure(figsize=(14, 10))
        
        #la heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt='.2f', 
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={'shrink': 0.8},
                    annot_kws={'size': 8})
        
        plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Heatmap Saved: {save_path}")
        
        plt.show()
    
    def plot_scatter(self, var1, var2, save_path=None):

        if self.data is None:
            self.load_data()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        #scatter plot
        ax.scatter(self.data[var1], self.data[var2], alpha=0.5, s=10, c='steelblue')
        
        # Ligne de régression
        z = np.polyfit(self.data[var1], self.data[var2], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.data[var1].min(), self.data[var1].max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear Regression')
        
        # Calcul de la corrélation
        coef, _ = self.calculate_correlation_pair(var1, var2)
        
        ax.set_xlabel(self.get_LABELS(var1), fontsize=11)
        ax.set_ylabel(self.get_LABELS(var2), fontsize=11)
        ax.set_title(f'Correlation: {self.get_label(var1)} vs {self.get_label(var2)} (r = {coef:.3f})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Scatter plot saved: {save_path}")
        
        plt.show()
    
    def plot_multiple_scatter(self, pairs, save_path=None):
        #crée plusieurs scatter plots pour différentes paires de variables.
    
        if self.data is None:
            self.load_data()
        
        n_pairs = len(pairs)
        cols = min(3, n_pairs)
        rows = (n_pairs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (var1, var2) in enumerate(pairs):
            ax = axes[idx]
            ax.scatter(self.data[var1], self.data[var2], alpha=0.4, s=8, c='steelblue')
            
            #régression
            z = np.polyfit(self.data[var1], self.data[var2], 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.data[var1].min(), self.data[var1].max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=1.5)
            
            #corrélation
            coef = self.data[var1].corr(self.data[var2])
            
            ax.set_xlabel(self.get_label(var1), fontsize=9)
            ax.set_ylabel(self.get_label(var2), fontsize=9)
            ax.set_title(f'r = {coef:.3f}', fontsize=10)

            ax.grid(True, alpha=0.3)
        
        #cacher les axes inutilisés
        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Scatter Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure Saved: {save_path}")
        
        plt.show()
    
    def interpret_correlation(self, coef):
 
        #Interprète la force d'une corrélation.
        abs_coef = abs(coef)
        if abs_coef >= 0.9:
            strength = "Very Strong"
        elif abs_coef >= 0.7:
            strength = "Strong"
        elif abs_coef >= 0.5:
            strength = "Moderate"
        elif abs_coef >= 0.3:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        direction = "positive" if coef > 0 else "negative"
        return f"{strength} correlation {direction}"


def test_correlation_analysis():
    
    print("=" * 60)
    print("Correlation Analysis Test")
    print("=" * 60)
    
    analyzer = CorrelationAnalyzer()
    
    print("\n1. Loading Data..")
    analyzer.load_data()

    print("\n2. Calculating Pearson Correlations.")
    pearson_matrix = analyzer.calculate_pearson_correlation()
 
    print("\n3. Calculating Spearman Correlations..")
    spearman_matrix = analyzer.calculate_spearman_correlation()
    
    print("\n4. Temperature vs Humidity Correlation..")
    coef, pvalue = analyzer.calculate_correlation_pair('temperature', 'humidity')
    print(f" Interpretation: {analyzer.interpret_correlation(coef)}")

    print("\n5. Top 10 Strongest Correlations..")
    top_corr = analyzer.get_strongest_correlations(n=10, method='pearson')
    
    print("\n6. Storing Results in Database..")
    analyzer.store_correlation_results(method='pearson')
    analyzer.store_correlation_results(method='spearman')
    
    print("\n7. Creating Heatmap..")
    analyzer.plot_heatmap(method='pearson', save_path='images/correlation_heatmap.png')
    
    print("\n8. Creating Scatter Plots..")
    pairs = [
    ('temperature', 'humidity'),
    ('temperature', 'co_gt'),
    ('co_gt', 'no2_gt'),
    ('temperature', 'no2_gt'),
    ('humidity', 'co_gt')
]
    analyzer.plot_multiple_scatter(pairs, save_path='images/scatter_plots.png')
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_correlation_analysis()
