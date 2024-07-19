"""line_follower controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import numpy as np
import joblib
import pandas as pd
import pickle
import river
import sys




# UP TO NOW WE USE 7 IR SENSORS
def run_robot(robot):
    time_step = 16
    max_speed = 6.28


    
    
    # initialize motors     
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
    
    
    irs = []
    
    # initialize sensors
    for i in range(0, 7):
        sensor = robot.getDevice('ir_{}'.format(i))
        sensor.enable(time_step)
        irs.append(sensor)
        
    
    
    while robot.step(time_step) != -1:
        
        irs_values = []
        for i in range(0,7):
            irs_values.append(irs[i].getValue())
        
        
        
        left_speed = max_speed * 0.7
        right_speed = max_speed * 0.7
        
        
        couple = {
            'straight': 0,
            'right': 2,
            'left': 1
        }
        loaded_model = joblib.load("C:\\Users\\franc\\Desktop\\TESI\\webots\\models\\model_decision_tree.joblib")
        
                
        df_pred = pd.DataFrame([irs_values], columns=[f'ir_{x}' for x in range(0,7)])
        # Make predictions with the loaded model
        prediction = loaded_model.predict(df_pred)
        
        if prediction[0] == 1:
            left_speed = -max_speed * 0.7
            
        elif prediction[0] == 2:
            right_speed = -max_speed * 0.7
            
        
        
            
        
        
        
        
        
        
        
        # Imposta le velocità dei motori
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
        
        
        

    
if __name__ == "__main__":
    my_robot = Robot()
    run_robot(my_robot)

