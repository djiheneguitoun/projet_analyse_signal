"""
Partie 4: Analyse Spectrale
============================
Ce module applique la Transformée de Fourier Rapide (FFT) sur les séries
temporelles pour identifier des tendances et oscillations périodiques.

Fonctionnalités:
- Application de la FFT sur les séries temporelles
- Identification des fréquences dominantes
- Stockage des résultats dans la base de données
- Visualisation des spectres de puissance
"""

import pandas as pd
import numpy as np
from scipy import fft
from scipy.signal import periodogram, welch
import matplotlib.pyplot as plt
from database_integration import AirQualityDatabase
from data_processing import DataProcessor


class SpectralAnalyzer:
    
    def __init__(self, db_path="air_quality.db"):

        self.db = AirQualityDatabase(db_path)
        self.data = None
        self.sampling_rate = 1.0  # 1 échantillon par heure
    
    def load_data(self):
    
        self.db.connect()
        self.data = self.db.get_data_as_dataframe()
        self.db.disconnect()
        
        print(f"Data Loaded: {len(self.data)} Records")
        return self.data
    
    def apply_fft(self, column):

        if self.data is None:
            self.load_data()
        
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' Not Found.")
        
        #récupérer les données et supprimer les NaN
        signal = self.data[column].dropna().values
        n = len(signal)
        
        #soustraire la moyenne (composante DC)
        signal = signal - np.mean(signal)
        
        # Appliquer la FFT
        fft_result = fft.fft(signal)
        
        # Calculer les fréquences
        frequencies = fft.fftfreq(n, d=1/self.sampling_rate)
        
        #garder que les fréquences positives
        positive_mask = frequencies >= 0
        frequencies = frequencies[positive_mask]
        fft_result = fft_result[positive_mask]
        
        # Calcul l'amplitude et la phase
        amplitudes = np.abs(fft_result) * 2 / n
        phases = np.angle(fft_result)
        
        print(f"FFT Applied on '{column}'")
        print(f"  - Points: {n}")
        print(f"  - Max Frequency: {frequencies[-1]:.4f} Hz")
        
        return frequencies, amplitudes, phases
    
    def compute_power_spectrum(self, column, method='periodogram'):

        if self.data is None:
            self.load_data()
        
        signal = self.data[column].dropna().values
        signal = signal - np.mean(signal)
        
        if method == 'periodogram':
            frequencies, power = periodogram(signal, fs=self.sampling_rate)
        else:  # welch
            frequencies, power = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
        
        print(f"Power Spectrum Calculated ({method}) for '{column}'")
        return frequencies, power
    
    def find_dominant_frequencies(self, column, n_peaks=5):

        frequencies, amplitudes, _ = self.apply_fft(column)
        
        #exclure la fréquence 0 (composante DC)
        mask = frequencies > 0
        frequencies = frequencies[mask]
        amplitudes = amplitudes[mask]
        
        #trouver les indices des pics les plus grands
        peak_indices = np.argsort(amplitudes)[-n_peaks:][::-1]
        
        results = []
        for idx in peak_indices:
            freq = frequencies[idx]
            amp = amplitudes[idx]
            period_hours = 1 / freq if freq > 0 else np.inf
            period_days = period_hours / 24
            
            results.append({
                'Frequency (Hz)': freq,
                'Amplitude': amp,
                'Period (hours)': period_hours,
                'Period (days)': period_days
            })
        
        df = pd.DataFrame(results)
        
        print(f"\n Top {n_peaks} Dominant Frequencies for '{column}':")
        print(df.to_string(index=False))
        
        return df
    
    def store_spectral_results(self, column):

        frequencies, power = self.compute_power_spectrum(column)
        dominant = self.find_dominant_frequencies(column, n_peaks=1)

        dominant_freq = dominant['Frequency (Hz)'].iloc[0]
        
        #convertir le spectre en string JSON pour stockage
        spectrum_data = {
            'frequencies': frequencies[:100].tolist(),  # Limiter la taille
            'power': power[:100].tolist()
        }
        
        self.db.connect()
        
        #supprimer les anciens résultats pour cette variable
        self.db.cursor.execute(
            "DELETE FROM spectral_analysis WHERE variable_name = ?", (column,)
        )
        
        #insérer les nouveaux résultats
        self.db.cursor.execute('''
            INSERT INTO spectral_analysis 
            (variable_name, dominant_frequency, power_spectrum_data)
            VALUES (?, ?, ?)
        ''', (column, float(dominant_freq), str(spectrum_data)))
        
        self.db.connection.commit()
        self.db.disconnect()
        
        print(f"Spectral Results Stored for '{column}'")
    
    def plot_fft_spectrum(self, column, save_path=None):
        
        frequencies, amplitudes, _ = self.apply_fft(column)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        #signal temporel (premiers 500 points)
        signal = self.data[column].dropna().values[:500]
        axes[0].plot(signal, 'b-', linewidth=0.8)
        axes[0].set_title(f'Time sinal: {column}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time (hours)')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Spectre FFT
        #limiter aux fréquences significatives
        mask = (frequencies > 0) & (frequencies < 0.1)
        axes[1].plot(frequencies[mask], amplitudes[mask], 'r-', linewidth=0.8)
        axes[1].set_title(f'FFT Spectrum: {column}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure Saved: {save_path}")
        
        plt.show()
    
    def plot_power_spectrum(self, column, method='welch', save_path=None):

        frequencies, power = self.compute_power_spectrum(column, method)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        #convertir fréquence en période (heures)
        periods = 1 / frequencies[1:]  # Éviter division par 0
        power_subset = power[1:]
        
        ax.semilogy(frequencies[1:], power_subset, 'b-', linewidth=0.8)
        ax.set_title(f'Power Spectrum ({method}): {column}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.grid(True, alpha=0.3, which='both')
        
        #ajouter les périodes caractéristiques
        ax.axvline(x=1/24, color='r', linestyle='--', alpha=0.7, label='Daily Cycle (24h)')
        ax.axvline(x=1/168, color='g', linestyle='--', alpha=0.7, label='Weekly Cycle (168h)')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure Saved: {save_path}")
        
        plt.show()
    
    def plot_multiple_spectra(self, columns, save_path=None):
  
        #Compare les spectres de plusieurs variables.
 
        if self.data is None:
            self.load_data()
        
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 2, figsize=(14, 4*n_cols))
        
        for idx, column in enumerate(columns):
            signal = self.data[column].dropna().values
            signal = signal - np.mean(signal)
            
            #signal temporel
            axes[idx, 0].plot(signal[:500], 'b-', linewidth=0.5)
            axes[idx, 0].set_title(f'{column} - Signal', fontsize=10)
            axes[idx, 0].set_xlabel('Time (h)')
            axes[idx, 0].grid(True, alpha=0.3)
            
            #spectrum
            frequencies, power = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
            axes[idx, 1].semilogy(frequencies, power, 'r-', linewidth=0.8)
            axes[idx, 1].set_title(f'{column} - Spectrum', fontsize=10)
            axes[idx, 1].set_xlabel('Frequency (Hz)')
            axes[idx, 1].axvline(x=1/24, color='g', linestyle='--', alpha=0.5)
            axes[idx, 1].grid(True, alpha=0.3, which='both')
        
        plt.suptitle('Multivariable Spectral Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure Saved: {save_path}")
        
        plt.show()
    
    def interpret_frequency(self, freq):
 
        #Interprète une fréquence en termes de cycle temporel.
        if freq <= 0:
            return "Composante constante"
        
        period_hours = 1 / freq
        
        if abs(period_hours - 24) < 2:
            return "Daily Cycle (~24h)"
        elif abs(period_hours - 12) < 1:
            return "Semi-Daily Cycle (~12h)"
        elif abs(period_hours - 168) < 10:
            return "Weekly Cycle (~7 jours)"
        elif period_hours > 500:
            return f"Long-Term Trend ({period_hours/24:.1f} days)"
        else:
            return f"Cycle of {period_hours:.1f} hours"


def test_spectral_analysis():

    print("=" * 60)
    print("SPECTRAL ANALYSIS TEST")
    print("=" * 60)
    
    analyzer = SpectralAnalyzer()
    
    print("\n1. Loading Data..")
    analyzer.load_data()
    
    print("\n2. Applying FFT on 'Temperature'..")
    freq, amp, phase = analyzer.apply_fft('temperature')
  
    print("\n3. Searching for Dominant Frequencies..")
    dominant_temp = analyzer.find_dominant_frequencies('temperature', n_peaks=5)
    
    print("\n Cycle Interpretation:")
    for _, row in dominant_temp.iterrows():
        interp = analyzer.interpret_frequency(row['Frequency (Hz)'])
        print(f"   - {interp}")
    
    print("\n4. Spectral Analysis of CO..")
    dominant_co = analyzer.find_dominant_frequencies('co_gt', n_peaks=5)
    
    print("\n5. Storing Results in the Database..")
    analyzer.store_spectral_results('temperature')
    analyzer.store_spectral_results('co_gt')
    analyzer.store_spectral_results('humidity')
    analyzer.store_spectral_results('no2_gt')
   
    print("\n6. Creating FFT Plot..")
    analyzer.plot_fft_spectrum('temperature', save_path='images/fft_temperature.png')
    
    print("\n7. Creating Power Spectrum..")
    analyzer.plot_power_spectrum('temperature', method='welch', save_path='images/power_spectrum.png')
   
    print("\n8. Multivariable Spectral Comparison..")
    analyzer.plot_multiple_spectra(
        ['temperature', 'humidity', 'co_gt', 'no2_gt'],
        save_path='images/spectral_comparison.png'
    )
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_spectral_analysis()
