from controller import Supervisor
import math

TIME_STEP = 16

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2)

def run_supervisor():
    supervisor = Supervisor()
    e_puck = supervisor.getFromDef('my_epuck')
    translation_field = e_puck.getField('translation')
    rotation_field = e_puck.getField('rotation')
    i = 0
    threshold = 0.015
    lap_completed = False

    while supervisor.step(TIME_STEP) != -1 and not lap_completed:
        if i == 0:
            start_position = translation_field.getSFVec3f()
            #start_rotation = rotation_field.getSFRotation()
            with open(r'C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\test_circuit\start_position.txt', 'w') as file:
                file.write(f"{start_position[0]},{start_position[1]},{start_position[2]}")
            #with open(r'C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\test_circuit\start_rotation.txt', 'w') as file:
            #    file.write(f"{start_rotation[0]},{start_rotation[1]},{start_rotation[2]},{start_rotation[3]}")
            with open(r'C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\test_circuit\lap_completed.txt', 'w') as file:
                file.write('0')

        current_position = translation_field.getSFVec3f()

        if distance(current_position, start_position) < threshold and i > 50:
            lap_completed = True
            print("Lap completed!")
            with open(r'C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\test_circuit\lap_completed.txt', 'w') as file:
                file.write('1')
              

        i += 1

if __name__ == "__main__":
    run_supervisor()
