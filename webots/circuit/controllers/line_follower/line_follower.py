"""line_follower controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import datetime
import numpy as np
from pathlib import Path




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
    
    
  
    sensors = []
    # initialize sensors
    for i in range(0, 7):
        sensor = robot.getDevice('ir_{}'.format(i))
        sensor.enable(time_step)
        sensors.append(sensor)
        
    
    sensor_data = [] 
    true_labels = []
    while robot.step(time_step) != -1:

        irs_values = []
        for i in range(0,7):
            irs_values.append(sensors[i].getValue())
        
        
        print(irs_values)
        left_speed = max_speed * 0.6
        right_speed = max_speed * 0.6
        
        # Determina quali sensori rilevano la linea
        line_not_detected = np.array([value >= 100 or value <= 5 for value in irs_values])
        
        
              
        weights = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
      
        somma = np.multiply(weights, ~line_not_detected)
        if sum(somma) < -0.01:
            left_speed = -max_speed * 0.6
            irs_values.append(1)
            sensor_data.append(irs_values)
            true_labels.append(1)
        elif sum(somma) > 0.01:
            right_speed = -max_speed * 0.6
            irs_values.append(2)
            sensor_data.append(irs_values)
            true_labels.append(2)
        else:
            irs_values.append(0)
            sensor_data.append(irs_values)
            true_labels.append(0)

        # Imposta le velocità dei motori
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)
        
    #save data to have true label
    file_name = f'true_label{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    path = Path('C:\\Users\\franc\\Desktop\\TESI\\webots\\training_data').joinpath(file_name)
    print(f'Saving data to{path}')
    np.savetxt(str(path), np.array(true_labels), delimiter=',') 
        
    #save data to train ml model
    file_name = f'sensor_data_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'
    path = Path('C:\\Users\\franc\\Desktop\\TESI\\webots\\training_data').joinpath(file_name)
    print(f'Saving data to{path}')
    #np.savetxt(str(path), np.array(sensor_data), delimiter=',')
    
    
if __name__ == "__main__":
    my_robot = Robot()
    run_robot(my_robot)




        
    