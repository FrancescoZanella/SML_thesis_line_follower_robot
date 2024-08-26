"""complete controller."""

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





TIME_STEP = 16
MAX_SPEED = 6.28
LFM_FORWARD_SPEED = 200
LFM_K_GS_SPEED = 0.4
NB_GROUND_SENS = 3

def initialize_devices(robot):

    # initialize motors     
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    #initialize camera
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
    
    return pretrained_model,metrics.Accuracy()


    
def load_left_velocity(irs_values):

    DeltaS = irs_values[2] - irs_values[0]  # destra - sinistra
     
    speed_l = LFM_FORWARD_SPEED - LFM_K_GS_SPEED * DeltaS
   
    return speed_l

def create_directories():

    # Ensure that the directories for data and plots exist
    base_path = Path(MODEL_PATH).parent.parent
    data_path = base_path.joinpath('data')
    sensors_data_path = data_path.joinpath('sensors_data')
    plots_path = base_path.joinpath('plots')
    images_path = base_path.joinpath('images')

    # Create directories if they don't exist
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

    sensors,left_motor,right_motor,camera = initialize_devices(robot)
    
    if PRODUCTION == 'True':
        pretrained_model,metric = load_model()

    create_directories()

    
    while robot.step(TIME_STEP) != -1:

             
        
        X,irs_values,image = get_sensors_data(sensors=sensors,camera=camera)

        if SAVE_IMAGES == 'True':
            camera.saveImage(str(Path(MODEL_PATH).parent.parent.joinpath('images').joinpath(f"image{i}.jpg")),100)
        
        sensors_data.append(irs_values)
        
        
        if PRODUCTION == 'True':
            
            vel = pretrained_model.predict_one(X)

            vel_true = load_left_velocity(irs_values=irs_values)
            
            labels.append(vel_true)
            
            metric.update(vel_true, vel)

            if VERBOSE == 'True':
                print(f'Predicted vel: {vel}')
                print(f'True vel: {vel_true}')

            pretrained_model.learn_one(X, vel_true)
            
            if VERBOSE == 'True':
                print(f'Accuracy: {metric.get()}')
            
            accuracy_log.append(metric.get())


        else:
            # move the robot using the "pid" way
            print(X)
            vel = load_left_velocity(irs_values=irs_values)
            labels.append(vel)

        
        left_motor.setVelocity(0.00628 * vel)
        right_motor.setVelocity(0.00628 * (400-vel))

        
    
    if SAVE_SENSORS == 'True' and PRODUCTION == 'False':
        
        for i in range(len(sensors_data)):
            sensors_data[i].append(labels[i])

        sensors_data = pd.DataFrame(sensors_data,columns=['sensor0','sensor1','sensor2','target'])
        path = Path(MODEL_PATH).parent.parent.joinpath('data').joinpath('sensors_data')
        file_name = f'sensor_data_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
        sensors_data.to_csv(str(path.joinpath(file_name)),index=False)
    
    
    if PLOT == 'True' and PRODUCTION == 'True':
        file_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
        plt.figure(figsize=(10, 5))
        plt.plot(accuracy_log, label='Accuracy')
        plt.xlabel('Time Step')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(MODEL_PATH).parent.parent.joinpath('plots').joinpath(file_name))
        
     
        
    
    
if __name__ == "__main__":

    MODEL_PATH = 'C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\models\\'

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




