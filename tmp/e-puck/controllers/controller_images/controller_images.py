from controller import Robot, Camera
from river import metrics
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import csv
import re
from collections import deque
import math
import numpy as np
from drift_detector import DriftDetector
from datetime import datetime
import pandas as pd
import tensorflow as tf
from tensorflow import keras as tfk
from PIL import Image
import io

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
    
    return pretrained_model,metrics.MAE()

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

def check_lost_track(irs_values):
    irs_values = np.array(irs_values)
    if irs_values.std() / irs_values.mean() * 100 < 10:
        return True
    else:
        return False
    
def img_to_emb(image, camera):
    model = tfk.models.load_model(r"C:\Users\franc\Desktop\prova.keras")
    
    # Get image dimensions
    width = camera.getWidth()
    height = camera.getHeight()
    
    # Convert raw bytes to numpy array
    np_image = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    
    # Remove alpha channel if present
    if np_image.shape[2] == 4:
        np_image = np_image[:, :, :3]
    
    # Resize the image
    np_image = tf.image.resize(np_image, (48, 48))
    
    # Normalize the image
    np_image = tf.cast(np_image, tf.float32) / 255.0  
    
    flatten_layer = tf.keras.Sequential(model.layers[:8])
    
    embedding = flatten_layer(tf.expand_dims(np_image, axis=0))
    
    embedding = tf.squeeze(embedding, axis=0).numpy()
    features_dict = {f'embedding_{i}': value for i, value in enumerate(embedding)}
    return features_dict


def run_robot(robot):
    mae_log = []
    labels = []
    sensors_data = []
    lost_track = deque(maxlen=10)
    last_velocities = deque(maxlen=10)
    valore_nero = 300
    valore_bianco = 850
    tolleranza = 50
    min_outlier_duration = 20
    min_gap_duration = 25 
    
    sensors, left_motor, right_motor,camera = initialize_devices(robot)

    if PRODUCTION == 'True':
        pretrained_model, metric = load_model()

    create_directories()

    i = 0
    recover_track = False
    last_is_drift = False
    

    drift_detector = DriftDetector(valore_nero, valore_bianco, tolleranza, min_outlier_duration, min_gap_duration)

    while robot.step(TIME_STEP) != -1:
        X, irs_values, image = get_sensors_data(sensors=sensors, camera=camera)
        features = img_to_emb(image, camera)
        X = {**X, **features}
        
        if SAVE_IMAGES == 'True':
            camera.saveImage(str(Path(MODEL_PATH).parent.parent.joinpath('images').joinpath(f"image{i}.jpg")),100)
        
        sensors_data.append(irs_values)
        drift_detector.update(irs_values[0])
        if PRODUCTION == 'True':
            if drift_detector.drift_detected and last_is_drift:
                vel_pred = model.predict_one(X)
            else:
                vel_pred = pretrained_model.predict_one(X)
            vel_true = load_left_velocity(irs_values)

            labels.append(vel_true)
            metric.update(vel_true, vel_pred)
            if LEARNING == 'True' and drift_detector.drift_detected and not last_is_drift:
                print('new model created')
                model = load_model()[0]
                last_is_drift = True
            if LEARNING == 'True' and drift_detector.drift_detected and last_is_drift:
                #print('learning') 
                
                model.learn_one(X, vel_true)
            if LEARNING == 'True' and not drift_detector.drift_detected and last_is_drift:
                last_is_drift = False
                del model
            if VERBOSE == 'True':
                print(f'MAE: {metric.get()}')
                
            
            vel = vel_pred
        else:
            vel = load_left_velocity(irs_values)
            labels.append(vel)

        
        lost_track.append(check_lost_track(irs_values))
        
        if all(lost_track):
            recover_track = True
        elif recover_track and not any(lost_track):
               recover_track = False
            
        if PRODUCTION == 'True':
            mae_log.append(metric.get())
        
        
        if recover_track and ENABLE_RECOVERY == 'True':
            print(f'Recovering track {i}')
            left_speed = max(min(0.00628 * (400 - np.mean(last_velocities)), MAX_SPEED), -MAX_SPEED)
            right_speed = max(min(0.00628 * np.mean(last_velocities), MAX_SPEED), -MAX_SPEED)
        else:
            left_speed = max(min(0.00628 * vel, MAX_SPEED), -MAX_SPEED)
            right_speed = max(min(0.00628 * (400 - vel), MAX_SPEED), -MAX_SPEED)

        last_velocities.append(left_speed)

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
        i += 1
        

    if PLOT == 'True':
        drift_detector.plot_anomalies(pd.DataFrame([x[0] for x in sensors_data], columns=['sensor0']))

        plt.figure(figsize=(12, 6))
        plt.plot(mae_log, label='MAE')
        plt.title('MAE over time with Drift Detection')
        plt.xlabel('Time steps')
        plt.ylabel('MAE')
        
        # Create a red background for drift intervals
        for start, end in drift_detector.anomalies:
            plt.axvspan(start, end, facecolor='red', alpha=0.2)
        
        # Plot MAE on top of the background
        plt.plot(mae_log, label='MAE', zorder=10)
        
        plt.legend()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        if LEARNING == 'True':
            name = f'rmse_drift_plot_learning_{current_time}.png'
        else:
            name = f'rmse_drift_plot_no_learning_{current_time}.png'
        plot_path = str(Path(MODEL_PATH).parent.parent.joinpath('plots', name))
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
    global MODEL_PATH, PRODUCTION, PLOT, SAVE_SENSORS, LEARNING, VERBOSE, ENABLE_RECOVERY, SAVE_IMAGES

    MODEL_PATH = Path(__file__).parent.parent.parent / 'data' / 'models'
    PRODUCTION = re.search(r"(?<=:\s).*$", sys.argv[1]).group(0)
    MODEL_PATH = str(MODEL_PATH.joinpath(re.search(r"(?<=:\s).*$", sys.argv[2]).group(0) + '.pkl'))
    PLOT = re.search(r"(?<=:\s).*$", sys.argv[3]).group(0)
    SAVE_SENSORS = re.search(r"(?<=:\s).*$", sys.argv[4]).group(0)
    SAVE_IMAGES = re.search(r"(?<=:\s).*$", sys.argv[5]).group(0)
    VERBOSE = re.search(r"(?<=:\s).*$", sys.argv[6]).group(0)
    LEARNING = re.search(r"(?<=:\s).*$", sys.argv[7]).group(0)
    ENABLE_RECOVERY = re.search(r"(?<=:\s).*$", sys.argv[8]).group(0)

    print(f"PRODUCTION: {PRODUCTION}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"PLOT: {PLOT}")
    print(f"SAVE_SENSORS: {SAVE_SENSORS}")
    print(f"SAVE_IMAGES: {SAVE_IMAGES}")
    print(f"VERBOSE: {VERBOSE}")
    print(f"LEARNING: {LEARNING}")
    print(f"ENABLE_RECOVERY: {ENABLE_RECOVERY}")

    my_robot = Robot()
    run_robot(my_robot)

if __name__ == "__main__":
    main()

