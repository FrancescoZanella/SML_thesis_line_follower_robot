import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


class DualDriftDetector:
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

    def update(self, left_value, right_value):
        
        is_outlier = (left_value < (self.nero - self.tol)) or ((left_value > (self.nero + self.tol)) and (left_value < (self.bianco - self.tol))) or (left_value > (self.bianco + self.tol)) or (right_value < (self.nero - self.tol)) or ((right_value > (self.nero + self.tol)) and (right_value < (self.bianco - self.tol))) or (right_value > (self.bianco + self.tol))
        
        if is_outlier:
            self.outlier_count += 1
            if self.outlier_count == self.min_duration and self.anomaly_start is None:
                if self.current_index - self.min_duration + 1 <= self.last_anomaly_end + self.min_gap:
                    if self.anomalies: 
                        self.anomaly_start = self.anomalies[-1][0]
                        self.anomalies.pop()
                else:
                    self.anomaly_start = self.current_index - self.min_duration + 1
                    self.drift_detected = True
                self.last_notification = "start"
        else:
            if self.anomaly_start is not None:
                self.anomalies.append((self.anomaly_start, self.current_index - 1))
                self.last_anomaly_end = self.current_index - 1
                self.anomaly_start = None
                self.last_notification = "end"
            elif self.last_notification == "end" and self.current_index > self.last_anomaly_end + self.min_gap:
                self.last_notification = None
                self.drift_detected = False
            self.outlier_count = 0
        
        self.current_index += 1
        return self.drift_detected