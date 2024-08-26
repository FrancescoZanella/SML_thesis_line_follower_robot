# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
e-puck_line -- Base code for a practical assignment on behavior-based
robotics. When completed, the behavior-based controller should allow
the e-puck robot to follow the black line and recover its path afterwards.
"""

from controller import Robot

# Global defines
TIME_STEP = 32  # [ms]
NB_GROUND_SENS = 3


LFM_FORWARD_SPEED = 200
LFM_K_GS_SPEED = 0.4


# 3 IR ground color sensors
gs = [None] * NB_GROUND_SENS
gs_value = [0] * NB_GROUND_SENS



def main():
    global left_motor, right_motor
    
    # Initialize the Webots Robot instance
    robot = Robot()

    for i in range(NB_GROUND_SENS):
        gs[i] = robot.getDevice(f"gs{i}")
        gs[i].enable(TIME_STEP)

    # Motors
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    

    while robot.step(TIME_STEP) != -1:
        

        for i in range(NB_GROUND_SENS):
            gs_value[i] = gs[i].getValue()

        
        speed_l,speed_r = 0, 0

        DeltaS = gs_value[2] - gs_value[0]  # destra - sinistra
        

        speed_l = LFM_FORWARD_SPEED - LFM_K_GS_SPEED * DeltaS
        speed_r = 400 - speed_l

        
        
        
        # Set wheel speeds
        left_motor.setVelocity(0.00628 * speed_l)
        right_motor.setVelocity(0.00628 * speed_r)

if __name__ == "__main__":
    main()
