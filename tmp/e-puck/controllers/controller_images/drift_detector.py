import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


class DriftDetector:
    def __init__(self, nero, bianco, tol, min_duration, min_gap):
        self.nero = nero
        self.bianco = bianco
        self.tol = tol
        self.min_duration = min_duration
        self.min_gap = min_gap
        self.anomalies = []
        self.outlier_count = 0
        self.anomaly_start = None
        self.last_anomaly_end = -1
        self.current_index = 0
        self.last_notification = None
        self.drift_detected = False

    def update(self, value):
        is_outlier = (value < (self.nero - self.tol)) or ((value > (self.nero + self.tol)) and (value < (self.bianco - self.tol))) or (value > (self.bianco + self.tol))
        
        if is_outlier:
            self.outlier_count += 1
            if self.outlier_count == self.min_duration and self.anomaly_start is None:
                if self.current_index - self.min_duration + 1 <= self.last_anomaly_end + self.min_gap:
                    self.anomaly_start = self.anomalies[-1][0]
                    self.anomalies.pop()
                else:
                    self.anomaly_start = self.current_index - self.min_duration + 1
                    print(f"drift detected from {self.anomaly_start}")
                    self.drift_detected = True
                self.last_notification = "start"
        else:
            if self.anomaly_start is not None:
                self.anomalies.append((self.anomaly_start, self.current_index - 1))
                self.last_anomaly_end = self.current_index - 1
                self.anomaly_start = None
                self.last_notification = "end"
            elif self.last_notification == "end" and self.current_index > self.last_anomaly_end + self.min_gap:
                print(f"drift ends at {self.last_anomaly_end}")
                self.last_notification = None
                self.drift_detected = False
            self.outlier_count = 0
        
        self.current_index += 1
        return self.drift_detected   
    
    def plot_anomalies(self,df):
        print("drifts (start, end):", self.anomalies)

        plt.figure(figsize=(10, 5))
        plt.plot(df, label='sensor')
        for start, end in self.anomalies:
            plt.axvspan(start, end, color='red', alpha=0.3, label='Drift' if start == self.anomalies[0][0] else '')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Drifts')
        plt.legend()
        plt.grid(True)
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = Path(__file__).parent.parent.parent.joinpath('data', 'plots', f'drifts{current_time}.png')
        plt.savefig(img_path)