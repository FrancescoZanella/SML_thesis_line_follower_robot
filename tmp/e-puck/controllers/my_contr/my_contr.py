from controller import Robot
import numpy as np
import os
import csv
import math

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
    for i in range(3):
        sensor = robot.getDevice(f'gs{i}')
        sensor.enable(TIME_STEP)
        sensors.append(sensor)

    return sensors,left_motor,right_motor

def load_left_velocity(irs_values):
    DeltaS = irs_values[2] - irs_values[0]
    speed_l = 200 - 1e-6 * math.pow(DeltaS, 3)
    return speed_l



def check_lap_completed():
    try:
        with open(r"C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\tmp\e-puck\data\exchange\lap_completed.txt", 'r') as file:
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
        vel = load_left_velocity(irs_values)
        if not check_lap_completed():
            actions.append(vel)
        
        left_speed = max(min(0.00628 * vel, MAX_SPEED), -MAX_SPEED)
        right_speed = max(min(0.00628 * (400 - vel), MAX_SPEED), -MAX_SPEED)

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

    os.remove(r"C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\tmp\e-puck\data\exchange\lap_completed.txt")
    
    with open(r'C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\tmp\e-puck\data\exchange\actions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for action in actions:
            writer.writerow([action])

    
if __name__ == "__main__":
    run_robot()
