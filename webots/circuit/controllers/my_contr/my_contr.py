from controller import Robot
import numpy as np
import os
import csv

TIME_STEP = 16
MAX_SPEED = 6.28

def initialize_devices(robot):
    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

    sensors = []
    for i in range(0, 7):
        sensor = robot.getDevice(f'ir_{i}')
        sensor.enable(TIME_STEP)
        sensors.append(sensor)

    return sensors, left_motor, right_motor

def load_label(irs_values):
    line_not_detected = np.array([value >= 100 or value <= 5 for value in irs_values])
    weights = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    somma = np.multiply(weights, ~line_not_detected)
    if sum(somma) < -0.01:
        return 1  # left
    elif sum(somma) > 0.01:
        return 2  # right
    else:
        return 0  # straight

def control_robot(prediction, left_motor, right_motor):
    left_speed = MAX_SPEED * 0.6
    right_speed = MAX_SPEED * 0.6

    if prediction == 1:
        left_speed = -MAX_SPEED * 0.6
    elif prediction == 2:
        right_speed = -MAX_SPEED * 0.6

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

def check_lap_completed():
    try:
        with open(r"C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\test_circuit\lap_completed.txt", 'r') as file:
            content = file.read().strip()
            return content == '1'
    except FileNotFoundError:
        return False
    
def run_robot():
    robot = Robot()
    sensors, left_motor, right_motor = initialize_devices(robot)

    actions = []
    
    while robot.step(TIME_STEP) != -1:
        irs_values = [sensor.getValue() for sensor in sensors]
        y_pred = load_label(irs_values)
        if not check_lap_completed():
            actions.append(y_pred)
        control_robot(y_pred, left_motor, right_motor)

    os.remove(r"C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\test_circuit\lap_completed.txt")
    
    with open(r'C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\test_circuit\actions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for action in actions:
            writer.writerow([action])

    
if __name__ == "__main__":
    run_robot()
