from controller import Robot, Camera
import numpy as np
from river import metrics,drift
import pickle
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import sys
import csv
from PIL import Image
import re
import pandas as pd
from collections import deque
import psutil
import os
import statistics
import math

TIME_STEP = 32
MAX_SPEED = 6.28
LFM_FORWARD_SPEED = 200
LFM_K_GS_SPEED = 1e-6
NB_GROUND_SENS = 3

def initialize_devices(robot):
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    camera = robot.getDevice('camera')
    camera.enable(TIME_STEP)

    sensors = []
    for i in range(NB_GROUND_SENS):
        sensor = robot.getDevice(f'gs{i}')
        sensor.enable(TIME_STEP)
        sensors.append(sensor)

    return sensors,left_motor,right_motor,camera

def load_model():
    with open(f'{MODEL_PATH}', 'rb') as f:
        pretrained_model = pickle.load(f)
    
    return pretrained_model,metrics.RMSE()

def load_left_velocity(irs_values):
    DeltaS = irs_values[2] - irs_values[0]
    speed_l = LFM_FORWARD_SPEED - LFM_K_GS_SPEED * math.pow(DeltaS, 3)
    return speed_l

def create_directories():
    base_path = Path(MODEL_PATH).parent.parent
    data_path = base_path.joinpath('data')
    sensors_data_path = data_path.joinpath('sensors_data')
    plots_path = base_path.joinpath('plots')
    images_path = base_path.joinpath('images')

    data_path.mkdir(parents=True, exist_ok=True)
    sensors_data_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True,exist_ok=True)

def get_sensors_data(sensors,camera):
    sensor_data_t = []
    for i in range(NB_GROUND_SENS):
        sensor_data_t.append(sensors[i].getValue())

    s_dict = {f'sensor{j}': val for j,val in enumerate(sensor_data_t)}
    
    image = camera.getImage()
    
    return s_dict,sensor_data_t,image

def run_robot(robot):
    accuracy_log = []
    labels = []
    sensors_data = []
    drift_points = []
    
    sensors, left_motor, right_motor, camera = initialize_devices(robot)

    if PRODUCTION == 'True':
        pretrained_model, metric = load_model()

    create_directories()

    i = 0
    rmse_window = deque(maxlen=100)  
    rmse_threshold = 3
    drift_introduced = False
    drift_factor = 2
    start_drift_step = 1000
    learning_enabled = False
   

    while robot.step(TIME_STEP) != -1:
        X, irs_values, image = get_sensors_data(sensors=sensors, camera=camera)
        
        if PRODUCTION == 'True' and i >= start_drift_step:
            original_irs_values = irs_values.copy()
            irs_values = [val * drift_factor for val in irs_values]
            if not drift_introduced:
                print(f"Drift significativo introdotto al passo {i}")
                drift_introduced = True
        else:
            original_irs_values = irs_values

        sensors_data.append(irs_values)

        if SAVE_IMAGES == 'True':
            camera.saveImage(str(Path(MODEL_PATH).parent.parent.joinpath('images').joinpath(f"image{i}.jpg")), 100)
        
        if PRODUCTION == 'True':
            X_for_prediction = {f'sensor{j}': val for j, val in enumerate(irs_values)}
            vel_pred = pretrained_model.predict_one(X_for_prediction)
            vel_true = load_left_velocity(original_irs_values)
            
            current_rmse = metric.get()
            rmse_window.append(current_rmse)
            
            if len(rmse_window) > 50:  
                median_rmse = statistics.median(rmse_window)
                print(f'CURRENT RMSE: {current_rmse}')
                print(f'TO_OVERCOME: {median_rmse*rmse_threshold}')
                if current_rmse > median_rmse * rmse_threshold:
                    drift_points.append(i)
                    if not learning_enabled:
                        learning_enabled = True
                    print(f"Drift detected at step {i}")
                    
            labels.append(vel_true)
            metric.update(vel_true, vel_pred)
            if learning_enabled:
                print(f'LEARNING: {X_for_prediction}')
                pretrained_model.learn_one(X_for_prediction, vel_true)
            
            if VERBOSE == 'True':
                print(f'RMSE: {metric.get()}')
                
            accuracy_log.append(metric.get())

            vel = vel_pred
        else:
            vel = load_left_velocity(irs_values)
            labels.append(vel)

        left_speed = max(min(0.00628 * vel, MAX_SPEED), -MAX_SPEED)
        right_speed = max(min(0.00628 * (400 - vel), MAX_SPEED), -MAX_SPEED)

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
        i += 1

        if learning_enabled:
            pretrained_model.learn_one(X_for_prediction, vel_true)

    if PLOT == 'True':
        plt.figure(figsize=(12, 6))
        plt.plot(accuracy_log, label='RMSE')
        plt.title('RMSE over time with Drift Detection')
        plt.xlabel('Time steps')
        plt.ylabel('RMSE')
        
        for point in drift_points:
            plt.axvline(x=point, color='r', linestyle='--', label='Drift Detected' if point == drift_points[0] else '')
        
        plt.legend()
        plot_path = str(Path(MODEL_PATH).parent.parent.joinpath('plots', 'rmse_drift_plot.png'))
        plt.savefig(plot_path)
        plt.close()

    if SAVE_SENSORS == 'True' and PRODUCTION == 'False':
        sensors_data_path = Path(MODEL_PATH).parent.parent.joinpath('data', 'sensors_data', 'sensors_data.csv')
        sensors_data_path.parent.mkdir(parents=True, exist_ok=True)

        with open(sensors_data_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['sensor0', 'sensor1', 'sensor2', 'target'])

            for irs, vel in zip(sensors_data, labels):
                csv_writer.writerow(irs + [vel])

    

def main():
    global MODEL_PATH, PRODUCTION, PLOT, SAVE_SENSORS, SAVE_IMAGES, VERBOSE

    MODEL_PATH = 'C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\tmp\\e-puck\\data\\models\\'

    PRODUCTION = re.search(r"(?<=:\s).*$", sys.argv[1]).group(0)
    MODEL_PATH += re.search(r"(?<=:\s).*$", sys.argv[2]).group(0) + '.pkl'
    PLOT = re.search(r"(?<=:\s).*$", sys.argv[3]).group(0)
    SAVE_SENSORS = re.search(r"(?<=:\s).*$", sys.argv[4]).group(0)
    SAVE_IMAGES = re.search(r"(?<=:\s).*$", sys.argv[5]).group(0)
    VERBOSE = re.search(r"(?<=:\s).*$", sys.argv[6]).group(0)

    print(f"PRODUCTION: {PRODUCTION}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"PLOT: {PLOT}")
    print(f"SAVE_SENSORS: {SAVE_SENSORS}")
    print(f'SAVE_IMAGES: {SAVE_IMAGES}')
    print(f"VERBOSE: {VERBOSE}")

    my_robot = Robot()
    run_robot(my_robot)

if __name__ == "__main__":
    main()

