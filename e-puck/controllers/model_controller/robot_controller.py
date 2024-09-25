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
from drift_detector import DualDriftDetector
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Oespl import OESPL



class RobotController:
    TIME_STEP = 32
    MAX_SPEED = 6.28
    LFM_FORWARD_SPEED = 200
    NB_GROUND_SENS = 3
    

    def __init__(self, production, model_path, plot, save_sensors, save_images, verbose, learning, enable_recovery):
        self.production = production
        self.model_path = model_path
        self.plot = plot
        self.save_sensors = save_sensors
        self.save_images = save_images
        self.verbose = verbose
        self.learning = learning
        self.enable_recovery = enable_recovery
        self.mem = 0
        self.MAX_MEM = 2000
        self.decay_rate = -math.log(0.001) / self.MAX_MEM

        self.robot = Robot()
        self.sensors,self.camera,self.left_motor, self.right_motor = self.initialize_devices()
        self.create_directories()

        if self.production:
            self.pretrained_model, self.metric = self.load_model()

        self.drift_detector = DualDriftDetector(300, 850, 50, 20, 25)
        self.mae_log = []
        self.labels = []
        self.sensors_data = []
        self.lost_track = deque(maxlen=10)
        self.last_velocities = deque(maxlen=10)

    def initialize_devices(self):
        left_motor = self.robot.getDevice('left wheel motor')
        right_motor = self.robot.getDevice('right wheel motor')
        left_motor.setPosition(float('inf'))
        right_motor.setPosition(float('inf'))
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        camera = self.robot.getDevice('camera')
        camera.enable(self.TIME_STEP)

        sensors = []
        for i in range(self.NB_GROUND_SENS):
            sensor = self.robot.getDevice(f'gs{i}')
            sensor.enable(self.TIME_STEP)
            sensors.append(sensor)

        return sensors,camera, left_motor, right_motor

    def load_model(self):
        with open(f'{self.model_path}', 'rb') as f:
            pretrained_model = pickle.load(f)
        
        return pretrained_model, metrics.MAE()

    def create_directories(self):
        base_path = Path(self.model_path).parent.parent
        data_path = base_path.joinpath('data')
        sensors_data_path = data_path.joinpath('sensors_data')
        plots_path = base_path.joinpath('plots')
        images_path = base_path.joinpath('images')

        data_path.mkdir(parents=True, exist_ok=True)
        sensors_data_path.mkdir(parents=True, exist_ok=True)
        plots_path.mkdir(parents=True, exist_ok=True)
        images_path.mkdir(parents=True, exist_ok=True)

    def get_sensors_data(self):
        sensor_data_t = []
        for i in range(self.NB_GROUND_SENS):
            sensor_data_t.append(self.sensors[i].getValue())

        s_dict = {f'sensor{j}': val for j, val in enumerate(sensor_data_t)}
        image = self.camera.getImage()
        return s_dict, sensor_data_t, image
    
    def check_lost_track(self, irs_values):
        irs_values = np.array(irs_values)
        if irs_values.std() / irs_values.mean() * 100 < 10:
            return True
        else:
            return False

    def load_true_labels(self, irs_values,sensible):
        DeltaS = irs_values[2] - irs_values[0]
        if sensible:
            speed_l = self.LFM_FORWARD_SPEED - 1e-6 * math.pow(DeltaS, 3)
        else:
            speed_l = self.LFM_FORWARD_SPEED - 0.6 * math.pow(DeltaS, 1)
        return speed_l

    def run(self):
        i = 0
        recover_track = False
        last_is_drift = False

        while self.robot.step(self.TIME_STEP) != -1:
            X, irs_values,image = self.get_sensors_data()
            

            if self.save_images == 'True':
                self.camera.saveImage(str(Path(self.model_path).parent.parent.joinpath('images').joinpath(f"image{i}.jpg")),100)
        
            
            self.sensors_data.append(irs_values)

            self.drift_detector.update(irs_values[0],irs_values[2])
            
            if self.production:
                vel,last_is_drift = self.handle_production_mode(X, irs_values, last_is_drift)  
            else:
                vel = self.load_true_labels(irs_values,sensible=True)
                self.labels.append(vel)

            recover_track = self.update_motor_speeds(vel, i, recover_track)
            i += 1

        self.post_run_actions()

    def handle_production_mode(self, X, irs_values, last_is_drift):

        is_drift = self.drift_detector.drift_detected


        # predict phase
        if is_drift:
            if last_is_drift:
                vel_pred = self.model.predict_one(X)
                vel_true = self.load_true_labels(irs_values, sensible=False)
            else:
                print(f'Creating a new model')
                self.model = self.load_model()[0]
                self.metric = metrics.MAE()
                self.mem = 0
                vel_pred = self.model.predict_one(X)
                vel_true = self.load_true_labels(irs_values, sensible=False)
        else:
            vel_pred = self.pretrained_model.predict_one(X)
            vel_true = self.load_true_labels(irs_values, sensible=True)

        # update phase
        self.labels.append(vel_true)
        self.metric.update(vel_true, vel_pred)

        if self.verbose:
            print(f'MAE: {self.metric.get()}')

        # learn phase
        if self.learning:
            if is_drift:
                if self.mem <= self.MAX_MEM:
                    weight = 3 * math.exp(-self.decay_rate * self.mem)
                    self.model.learn_one(X, vel_true,sample_weight=weight)
                    self.mem += 1
            else:
                if last_is_drift:
                    print(f'Deleting model at step')
                    del self.model
                    self.metric = metrics.MAE()
                    self.mem = 0

        return vel_pred,is_drift

    

    def update_motor_speeds(self, vel, i, recover_track):
        self.lost_track.append(self.check_lost_track(self.sensors_data[-1]))

        if all(self.lost_track):
            recover_track = True
        elif recover_track and not any(self.lost_track):
            recover_track = False

        if self.production:
            self.mae_log.append(self.metric.get())

        if recover_track and self.enable_recovery:
            print(f'Recovering track {i}')
            left_speed = max(min(0.00628 * (400 - np.mean(self.last_velocities)), self.MAX_SPEED), -self.MAX_SPEED)
            right_speed = max(min(0.00628 * np.mean(self.last_velocities), self.MAX_SPEED), -self.MAX_SPEED)
        else:
            left_speed = max(min(0.00628 * vel, self.MAX_SPEED), -self.MAX_SPEED)
            right_speed = max(min(0.00628 * (400 - vel), self.MAX_SPEED), -self.MAX_SPEED)

        self.last_velocities.append(left_speed)

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        return recover_track

    def post_run_actions(self):
        if self.plot:
            current_time = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
            study_dir = f"study_{current_time}"
            plots_path = Path(self.model_path).parent.parent.joinpath('plots', study_dir)
            plots_path.mkdir(parents=True, exist_ok=True)
            
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
        
        

        
        plot_path = path.joinpath('mae_drift_plot.png')
        plt.savefig(plot_path)
        plt.close()

   
        

    def save_sensor_data(self):
        sensors_data_path = Path(self.model_path).parent.parent.joinpath('data', 'sensors_data', 'sensors_data.csv')
        sensors_data_path.parent.mkdir(parents=True, exist_ok=True)

        with open(sensors_data_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['sensor0', 'sensor1', 'sensor2', 'target'])

            for irs, vel in zip(self.sensors_data, self.labels):
                csv_writer.writerow(irs + [vel])


        


