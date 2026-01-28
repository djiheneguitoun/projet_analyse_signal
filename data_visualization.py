import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.signal import welch
from database_integration import AirQualityDatabase
import os

LABELS = {
    'co_gt': 'CO',
    'no2_gt': 'NO2',
    'temperature': 'Temperature',
    'humidity': 'Humidity'
}

class DataVisualizer:

    def __init__(self, db_path="air_quality.db"):

        self.db = AirQualityDatabase(db_path)
        self.data = None
        
        #configuration du style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    def load_data(self):

        self.db.connect()
        self.data = self.db.get_data_as_dataframe()
        self.db.disconnect()
        
        if 'date' in self.data.columns and 'time' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(
                self.data['date'] + ' ' + self.data['time'].str.replace('.', ':'),
                format='%d/%m/%Y %H:%M:%S',
                errors='coerce'
            )
        
        print(f"Data loaded: {len(self.data)} records")
        return self.data
    
    def plot_time_series(self, column, title=None, save_path=None):

        if self.data is None:
            self.load_data()
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        if 'datetime' in self.data.columns:
            x = self.data['datetime']
        else:
            x = range(len(self.data))
        
        ax.plot(x, self.data[column], color=self.colors[0], linewidth=0.5, alpha=0.8)
        
        #ajouter une moyenne mobile
        window = 24  # 24 heures
        rolling_mean = self.data[column].rolling(window=window, center=True).mean()
        ax.plot(x, rolling_mean, color=self.colors[1], linewidth=2, 
                label=f'Moving average ({window}h)')
        
        ax.set_xlabel('Date/Time', fontsize=11)
        ax.set_ylabel(column, fontsize=11)
        ax.set_title(title or f'Time Series: {column}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if 'datetime' in self.data.columns:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        plt.show()
    
    def plot_multiple_time_series(self, columns, save_path=None):

        if self.data is None:
            self.load_data()
        
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=(14, 3*n_cols), sharex=True)
        
        if n_cols == 1:
            axes = [axes]
        
        if 'datetime' in self.data.columns:
            x = self.data['datetime']
        else:
            x = range(len(self.data))
        
        for idx, col in enumerate(columns):
            axes[idx].plot(x, self.data[col], color=self.colors[idx % len(self.colors)], 
                          linewidth=0.5, alpha=0.7)
            
            #Moving average
            rolling_mean = self.data[col].rolling(window=24, center=True).mean()
            axes[idx].plot(x, rolling_mean, color='red', linewidth=1.5, alpha=0.8)
            
            axes[idx].set_ylabel(col, fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_title(col, fontsize=11, fontweight='bold', loc='left')
        
        axes[-1].set_xlabel('Date/Time', fontsize=11)
        
        if 'datetime' in self.data.columns:
            axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
        
        plt.suptitle('Time Series of Environmental Variables', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        plt.show()
    
    
    def plot_scatter(self, x_col, y_col, save_path=None):

        if self.data is None:
            self.load_data()
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        #scatter plot avec densité de couleur
        scatter = ax.scatter(self.data[x_col], self.data[y_col], 
                            c=self.data.index, cmap='viridis',
                            alpha=0.5, s=10)
        
        #ligne de régression
        mask = ~(self.data[x_col].isna() | self.data[y_col].isna())
        z = np.polyfit(self.data.loc[mask, x_col], self.data.loc[mask, y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(self.data[x_col].min(), self.data[x_col].max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear regression')
        
        #corrélation
        corr = self.data[x_col].corr(self.data[y_col])
        
        ax.set_xlabel(LABELS.get(x_col, x_col), fontsize=11)
        ax.set_ylabel(LABELS.get(y_col, y_col), fontsize=11)
        ax.set_title(f'{LABELS.get(x_col, x_col)} vs {LABELS.get(y_col, y_col)} (r = {corr:.3f})', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, label='Time index')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        plt.show()
    
    def plot_scatter_matrix(self, columns, save_path=None):

        if self.data is None:
            self.load_data()
        
        #Échantillonner pour performance
        sample = self.data[columns].dropna().sample(min(1000, len(self.data)))
        
        fig = plt.figure(figsize=(12, 12))
        
        # Utiliser seaborn pairplot
        g = sns.pairplot(sample, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 10},
                        diag_kws={'color': self.colors[0]})
        g.fig.suptitle('Scatter Matrix', fontsize=14, fontweight='bold', y=1.02)
        
        if save_path:
            g.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        plt.show()
    
    
    def plot_correlation_heatmap(self, columns=None, method='pearson', save_path=None):

        if self.data is None:
            self.load_data()
        
        if columns is None:
            columns = ['co_gt','no2_gt', 
                      'temperature', 'humidity']
        
        # Filtrer les colonnes disponibles
        columns = [c for c in columns if c in self.data.columns]
        
        corr_matrix = self.data[columns].corr(method=method)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, square=True,
                   linewidths=0.5, cbar_kws={'shrink': 0.8},
                   ax=ax)
        
        ax.set_title(f'Correlation Matrix ({method.capitalize()})', 
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        plt.show()
    
    def plot_temporal_heatmap(self, column, save_path=None):

        if self.data is None:
            self.load_data()
        
        if 'datetime' not in self.data.columns:
            print("Datetime column not available")
            return
        
        #extraire heure et jour de la semaine
        df = self.data.copy()
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        
        # Créer le pivot
        pivot = df.pivot_table(values=column, index='hour', 
                              columns='dayofweek', aggfunc='mean')
        
        #renommer les jours
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        pivot.columns = [day_names[i] for i in pivot.columns]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.1f', ax=ax)
        
        ax.set_xlabel('Day of the Week', fontsize=11)
        ax.set_ylabel('Hour', fontsize=11)
        ax.set_title(f'Temporal Distribution: {column}', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        plt.show()
    
    def plot_spectral_analysis(self, column, save_path=None):

        if self.data is None:
            self.load_data()
        
        signal = self.data[column].dropna().values
        signal = signal - np.mean(signal)
        
        # Spectre de Welch
        frequencies, power = welch(signal, fs=1.0, nperseg=min(256, len(signal)//4))
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Signal temporel
        axes[0].plot(signal[:1000], color=self.colors[0], linewidth=0.5)
        axes[0].set_title(f'Time Signal: {column}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time (hours)')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Spectre de puissance
        axes[1].semilogy(frequencies, power, color=self.colors[1], linewidth=1)
        axes[1].axvline(x=1/24, color='red', linestyle='--', alpha=0.7, label='24h')
        axes[1].axvline(x=1/12, color='green', linestyle='--', alpha=0.7, label='12h')
        axes[1].axvline(x=1/168, color='purple', linestyle='--', alpha=0.7, label='7 jours')
        axes[1].set_title(f'Power Spectrum: {column}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Spectral Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        plt.show()
    
    #IMAGES
    
    def display_processed_images(self, image_dir="images", save_path=None):
        import cv2
        
        image_files = {
            'Original': '1.png',
            'Grayscale': 'processed_grayscale.png',
            'Gaussian Blur': 'processed_blurred.png',
            'Edges (Canny)': 'processed_edges.png',
            'Otsu Thresholding': 'processed_threshold.png'
        }
        
        #vérifier les images disponibles
        available = {}
        for title, filename in image_files.items():
            path = os.path.join(image_dir, filename)
            if os.path.exists(path):
                available[title] = path
        
        if not available:
            print("No processed image found.")
            return
        
        n_images = len(available)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = np.array(axes).flatten() if n_images > 1 else [axes]
        
        for idx, (title, path) in enumerate(available.items()):
            img = cv2.imread(path)
            if img is not None:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[idx].imshow(img)
                else:
                    axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(title, fontsize=11, fontweight='bold')
            axes[idx].axis('off')
        
        for idx in range(len(available), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Processed Images', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved: {save_path}")
        
        plt.show()

def test_data_visualization():

    print("=" * 60)
    print("Data Visualization Test")
    print("=" * 60)
    
    viz = DataVisualizer()

    print("\n1. Loading data..")
    viz.load_data()

    print("\n2. Time Series Chart (Temperature)..")
    viz.plot_time_series('temperature', save_path='images/viz_timeseries_temp.png')
    
    print("\n3. Multiple Time Series..")
    viz.plot_multiple_time_series(
        ['temperature', 'humidity', 'co_gt'],
        save_path='images/viz_timeseries_multiple.png'
    )

    print("\n4. Scatter Plot..")
    viz.plot_scatter('temperature', 'humidity', save_path='images/viz_scatter.png')
    
    print("\n5. Correlation Heatmap..")
    viz.plot_correlation_heatmap(save_path='images/viz_heatmap.png')
    
    print("\n6. Temporal Heatmap..")
    viz.plot_temporal_heatmap('temperature', save_path='images/viz_temporal_heatmap.png')
    
    print("\n7. Spectral Analysis Visualization..")
    viz.plot_spectral_analysis('temperature', save_path='images/viz_spectral.png')
    
    print("\n8. Displaying Processed Images..")
    viz.display_processed_images(save_path='images/viz_images.png')
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_data_visualization()
