"""complete controller."""

from controller import Robot
import numpy as np
from river import metrics
import pandas as pd
import pickle
import matplotlib.pyplot as plt



MODEL_TO_LOAD = "arf"

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

    sensors = []
    # initialize sensors
    for i in range(0, 7):
        sensor = robot.getDevice('ir_{}'.format(i))
        sensor.enable(TIME_STEP)
        sensors.append(sensor)

    return sensors,left_motor,right_motor


def load_model():

    with open(f'C:\\Users\\franc\\Desktop\\TESI\\webots\\models\\model_{MODEL_TO_LOAD}.pkl', 'rb') as f:
        pretrained_model = pickle.load(f)
    
    return pretrained_model,metrics.Accuracy()



    
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

def get_sensors_data(sensors):
    sensor_data_t = []
    for i in range(0,7):
        sensor_data_t.append(sensors[i].getValue())

    
    s_dict = {f'ir_{j}': val for j,val in enumerate(sensor_data_t)}
    return s_dict,sensor_data_t

def control_robot(prediction,left_motor,right_motor,left_speed,right_speed):
    if prediction == 1:
        left_speed = -MAX_SPEED * 0.6
            
    elif prediction == 2:
        right_speed = -MAX_SPEED * 0.6
            
    # Imposta le velocità dei motori
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)   
        

    
    

        
        
        
        
        
        
        
        

def run_robot(robot):
    
    
    sensors,left_motor,right_motor = initialize_devices(robot)
    
    # load the model trained on classic environment
    pretrained_model,metric = load_model()
    
    
    
    accuracy_log = []
    
    while robot.step(TIME_STEP) != -1:
           

        
        couple = {
            0: 'straight',
            2: 'right',
            1: 'left'
        }

        left_speed = MAX_SPEED * 0.6
        right_speed = MAX_SPEED * 0.6
        
        # get sensors(new) data with drift inserted at some time
        X,irs_values = get_sensors_data(sensors=sensors)

        
        
        
        # use the model to predict how the model should move
        y_pred = pretrained_model.predict_one(X)
        y = load_label(irs_values)
        
        print(f'Predicted label: {couple[int(y_pred)]}')
        print(f'True label: {couple[int(y)]}')

        control_robot(y_pred,left_motor,right_motor,left_speed,right_speed)

            
        #update the model      
        pretrained_model.learn_one(X, y)    
        metric.update(y, y_pred)
        
        
        
        

        print(f'Accuracy: {metric.get()}')
        accuracy_log.append(metric.get())
        
    
        
    
        
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_log, label='Accuracy')
    plt.xlabel('Time Step')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
        
     
        
    
    
if __name__ == "__main__":
    my_robot = Robot()
    run_robot(my_robot)


"""
provare inclinazione metti features dell'inclinazione e ritraina modello
random forest funziona sicuro
hat
provare ad esempio a salvarsi e a ritornare 
cnn prendere il layer -1 di una cnn e farci featurese usare river
"""

