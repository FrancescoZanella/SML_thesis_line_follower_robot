from controller import Robot
import os

from collections import deque
import csv
from datetime import datetime
import math
from pathlib import Path
import pickle
from river import metrics
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Oespl import OESPL
from river import drift
from river import tree
from drift_detector import DualDriftDetector



class RobotControllerOESPL:
    TIME_STEP = 32
    MAX_SPEED = 6.28
    LFM_FORWARD_SPEED = 200
    NB_GROUND_SENS = 3
    

    def __init__(self, production, model_path, plot, save_sensors, verbose, learning):
        self.production = production
        self.model_path = model_path
        self.plot = plot
        self.save_sensors = save_sensors
        self.verbose = verbose
        self.learning = learning
        
        self.robot = Robot()
        self.sensors,self.left_motor, self.right_motor = self.initialize_devices()
        self.create_directories()

        if self.production:
            self.pretrained_model, self.metric = self.load_model()

        self.mae_log = []
        self.labels = []
        self.sensors_data = []
       

    def initialize_devices(self):
        left_motor = self.robot.getDevice('left wheel motor')
        right_motor = self.robot.getDevice('right wheel motor')
        left_motor.setPosition(float('inf'))
        right_motor.setPosition(float('inf'))
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        

        sensors = []
        for i in range(self.NB_GROUND_SENS):
            sensor = self.robot.getDevice(f'gs{i}')
            sensor.enable(self.TIME_STEP)
            sensors.append(sensor)

        return sensors, left_motor, right_motor

    def load_model(self):
        
        with open(f'{self.model_path}', 'rb') as f:
            pretrained_model = pickle.load(f)
        
        
        return OESPL(
            base_estimator=pretrained_model,
            ensemble_size=1,
            lambda_fixed=12,
            seed=42,
            drift_detector=DualDriftDetector(300, 850, 50, 20, 25),
            patience=500,
            awakening=100,
            reset_model=True),metrics.MAE()

    def create_directories(self):
        base_path = Path(self.model_path).parent.parent
        data_path = base_path.joinpath('data')
        sensors_data_path = data_path.joinpath('sensors_data')
        plots_path = base_path.joinpath('plots')

        data_path.mkdir(parents=True, exist_ok=True)
        sensors_data_path.mkdir(parents=True, exist_ok=True)
        plots_path.mkdir(parents=True, exist_ok=True)

    def get_sensors_data(self):
        sensor_data_t = []
        for i in range(self.NB_GROUND_SENS):
            sensor_data_t.append(self.sensors[i].getValue())

        s_dict = {f'sensor{j}': val for j, val in enumerate(sensor_data_t)}
        
        return s_dict, sensor_data_t
    
    

    def load_true_labels(self, irs_values,sensible):
        DeltaS = irs_values[2] - irs_values[0]
        if sensible:
            speed_l = self.LFM_FORWARD_SPEED - 1e-6 * math.pow(DeltaS, 3)
        else:
            speed_l = self.LFM_FORWARD_SPEED - 0.6 * math.pow(DeltaS, 1)
        return speed_l

    def run(self):
        i = 0
        while self.robot.step(self.TIME_STEP) != -1:
            
            X, irs_values = self.get_sensors_data()
            
            self.sensors_data.append(irs_values)

            if self.production:

                vel = self.pretrained_model.predict_one(X)
                vel_true = self.load_true_labels(irs_values, sensible=True)
        
                self.labels.append(vel_true)
                self.metric.update(vel_true, vel)

                if self.learning:
                    reset_metric = self.pretrained_model.learn_one(X, vel_true)
                    if reset_metric:
                        self.metric = metrics.MAE()
                    
                
                if self.verbose:
                    print(f'MAE: {self.metric.get()}')
            else:
                vel = self.load_true_labels(irs_values,sensible=True)
                self.labels.append(vel)

            self.update_motor_speeds(vel)
            i += 1
            

        self.post_run_actions()


   

    def update_motor_speeds(self, vel):
        

        if self.production:
            self.mae_log.append(self.metric.get())

        left_speed = max(min(0.00628 * vel, self.MAX_SPEED), -self.MAX_SPEED)
        right_speed = max(min(0.00628 * (400 - vel), self.MAX_SPEED), -self.MAX_SPEED)


        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        

    def post_run_actions(self):
        if self.plot:
            current_time = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
            study_dir = f"study_{current_time}"
            plots_path = Path(self.model_path).parent.parent.joinpath('plots', study_dir)
            plots_path.mkdir(parents=True, exist_ok=True)
            
            with open('prova.txt', 'w') as f:
                f.write(str(plots_path))

            self.plot_MAE_drift(plots_path)
        if self.save_sensors and not self.production:
            self.save_sensor_data()

    def plot_MAE_drift(self,path):
        plt.figure(figsize=(12, 6))
        plt.plot(self.mae_log, label='MAE', zorder=10)
        plt.title('MAE over time with Drift Detection')
        plt.xlabel('Time steps')
        plt.ylabel('MAE')
        
        for start, end in self.drift_detector.anomalies:
            plt.axvspan(start, end, facecolor='red', alpha=0.2, label='Concept')
            plt.axvline(x=start, color='red', linestyle='--', linewidth=1, label='Concept Drift')
            plt.axvline(x=end, color='red', linestyle='--', linewidth=1, label='Concept Drift')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')

    def save_sensor_data(self):
        sensors_data_path = Path(self.model_path).parent.parent.joinpath('data', 'sensors_data', 'sensors_data.csv')
        sensors_data_path.parent.mkdir(parents=True, exist_ok=True)

        with open(sensors_data_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['sensor0', 'sensor1', 'sensor2', 'target'])

            for irs, vel in zip(self.sensors_data, self.labels):
                csv_writer.writerow(irs + [vel])

    