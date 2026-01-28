import pandas as pd
import numpy as np
from scipy import fft
from scipy.signal import welch
from database_integration import AirQualityDatabase


class SpectralAnalyzer:
    
    def __init__(self, db_path="db_air_quality"):
        self.db = AirQualityDatabase(db_path)
        self.data = None
        self.sampling_rate = 1.0
    
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
        
        signal = self.data[column].dropna().values
        n = len(signal)
        
        signal = signal - np.mean(signal)
        
        fft_result = fft.fft(signal)
        
        frequencies = fft.fftfreq(n, d=1/self.sampling_rate)
        
        positive_mask = frequencies >= 0
        frequencies = frequencies[positive_mask]
        fft_result = fft_result[positive_mask]
        
        amplitudes = np.abs(fft_result) * 2 / n
        phases = np.angle(fft_result)
        
        print(f"FFT Applied on '{column}'")
        print(f"  - Points: {n}")
        print(f"  - Max Frequency: {frequencies[-1]:.4f} h-1")
        
        return frequencies, amplitudes, phases
    
    def compute_power_spectrum(self, column, method='periodogram'):
        if self.data is None:
            self.load_data()
        
        signal = self.data[column].dropna().values
        signal = signal - np.mean(signal)
        
        frequencies, power = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)//4))
        
        print(f"Power Spectrum Calculated ({method}) for '{column}'")
        return frequencies, power
    
    def find_dominant_frequencies(self, column, n_peaks=5):
        frequencies, amplitudes, _ = self.apply_fft(column)
        
        mask = frequencies > 0
        frequencies = frequencies[mask]
        amplitudes = amplitudes[mask]
        
        peak_indices = np.argsort(amplitudes)[-n_peaks:][::-1]
        
        results = []
        for idx in peak_indices:
            freq = frequencies[idx]
            amp = amplitudes[idx]
            period_hours = 1 / freq if freq > 0 else np.inf
            period_days = period_hours / 24
            
            results.append({
                'Frequency (h-1)': freq,
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

        dominant_freq = dominant['Frequency (h-1)'].iloc[0]
        
        spectrum_data = {
            'frequencies': frequencies[:100].tolist(),
            'power': power[:100].tolist()
        }
        
        self.db.connect()
        
        self.db.cursor.execute(
            "DELETE FROM spectral_analysis WHERE variable_name = %s", (column,)
        )
        
        self.db.cursor.execute('''
            INSERT INTO spectral_analysis 
            (variable_name, dominant_frequency, power_spectrum_data)
            VALUES (%s, %s, %s)
        ''', (column, float(dominant_freq), str(spectrum_data)))
        
        self.db.connection.commit()
        self.db.disconnect()
        
        print(f"Spectral Results Stored for '{column}'")

