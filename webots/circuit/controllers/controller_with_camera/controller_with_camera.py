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





TIME_STEP = 16
MAX_SPEED = 6.28

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
    # initialize sensors
    for i in range(0, 7):
        sensor = robot.getDevice('ir_{}'.format(i))
        sensor.enable(TIME_STEP)
        sensors.append(sensor)

    return sensors,left_motor,right_motor,camera


def load_model():
    with open(f'{MODEL_PATH}', 'rb') as f:
        pretrained_model = pickle.load(f)
    
    return pretrained_model,metrics.Accuracy()

def save_data(path, data, type):
    #save data to have true label
    file_name = f'{type} {datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    print(f'Saving data to {path}')
    np.savetxt(str(Path(path).joinpath(file_name)), np.array(data), delimiter=',')

def load_true_labels():
    with open('C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\data\\labels\\true_labels 2024-08-20_16-27-05.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        l = [float(riga[0]) for riga in csv_reader] 

    return l

def load_label(irs_values):
    line_not_detected = np.array([value >= 100 or value <= 5 for value in irs_values])
            
    weights = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
     
    somma = np.multiply(weights, ~line_not_detected)
    if sum(somma) < -0.01:
        return 1
    elif sum(somma) > 0.01:
        return 2
    else:
        return 0

def create_directories():

    # Ensure that the directories for data and plots exist
    base_path = Path(MODEL_PATH).parent.parent
    data_path = base_path.joinpath('data')
    labels_path = data_path.joinpath('labels')
    sensors_data_path = data_path.joinpath('sensors_data')
    plots_path = base_path.joinpath('plots')
    images_path = base_path.joinpath('images')

    # Create directories if they don't exist
    data_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)
    sensors_data_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True,exist_ok=True)

def get_sensors_data(sensors,camera):
    sensor_data_t = []
    for i in range(0,7):
        sensor_data_t.append(sensors[i].getValue())

    
    s_dict = {f'ir_{j}': val for j,val in enumerate(sensor_data_t)}
    
    image = camera.getImage()
    


    return s_dict,sensor_data_t,image

def control_robot(prediction,left_motor,right_motor,left_speed,right_speed):
    if prediction == 1:
        left_speed = -MAX_SPEED * 0.6
            
    elif prediction == 2:
        right_speed = -MAX_SPEED * 0.6
            
    # Imposta le velocità dei motori
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)   
        

    
    

        
        
        
        
        
        
        
        

def run_robot(robot):

    adwin = drift.ADWIN()
        
    accuracy_log = []
    labels = []
    sensors_data = []
    images = []

    sensors,left_motor,right_motor,camera = initialize_devices(robot)
    
    if PRODUCTION == 'True':
        # load the model trained on classic environment
        pretrained_model,metric = load_model()

    create_directories()

    #l = load_true_labels()
    i = 0
    update_frequency = 5

    
    while robot.step(TIME_STEP) != -1:
           

        couple = {
            0: 'straight',
            2: 'right',
            1: 'left'
        }

        left_speed = MAX_SPEED * 0.6
        right_speed = MAX_SPEED * 0.6
        
        # get sensors(new) data with drift inserted at some time
        X,irs_values,image = get_sensors_data(sensors=sensors,camera=camera)


        if SAVE_IMAGES == 'True':
            camera.saveImage(str(Path(MODEL_PATH).parent.parent.joinpath('images').joinpath(f"image{i}.jpg")),100)
        
        images.append(image)
        sensors_data.append(irs_values)
                
        
        if PRODUCTION == 'True':
            # use the model to predict how the model should move
            y_pred = pretrained_model.predict_one(X)

            y = l[i]
            #y = load_label(irs_values)
            labels.append(y)
            
            metric.update(y, y_pred)

            if VERBOSE == 'True':
                print(f'Predicted label: {couple[int(y_pred)]}')
                print(f'True label: {couple[int(y)]}')

            if i % update_frequency == 0:
                pretrained_model.learn_one(X, y)
            
            if VERBOSE == 'True':
                print(f'Accuracy: {metric.get()}')
            
            accuracy_log.append(metric.get())
        else:
            # move the robot using the "pid" way
            y_pred = load_label(irs_values)
            labels.append(y_pred)

               
        
        control_robot(y_pred,left_motor,right_motor,left_speed,right_speed)
        i += 1 
        
    

    if SAVE_LABELS == 'True' and PRODUCTION == 'False':
        # save labels
        save_data(Path(MODEL_PATH).parent.parent.joinpath('data').joinpath('labels'),labels,"true_labels")
    if SAVE_SENSORS == 'True' and PRODUCTION == 'False':
        # save data to train ml model
        for i in range(len(sensors_data)):
            sensors_data[i].append(labels[i])
        save_data(Path(MODEL_PATH).parent.parent.joinpath('data').joinpath('sensors_data'),sensors_data,"sensor_data")
    
    
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
    arg = sys.argv[1].split()
    
    PRODUCTION = arg[0]
    MODEL_PATH = arg[1]
    PLOT = arg[2]
    SAVE_SENSORS = arg[3]
    SAVE_LABELS = arg[4]
    SAVE_IMAGES = arg[5]
    VERBOSE = arg[6]


    print(f"PRODUCTION: {PRODUCTION}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"PLOT: {PLOT}")
    print(f"SAVE_SENSORS: {SAVE_SENSORS}")
    print(f"SAVE_LABELS: {SAVE_LABELS}")
    print(f'SAVE_IMAGES: {SAVE_IMAGES}')
    print(f"VERBOSE: {VERBOSE}")

    my_robot = Robot()

    run_robot(my_robot)




