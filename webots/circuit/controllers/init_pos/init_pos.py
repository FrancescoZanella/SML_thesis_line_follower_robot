from controller import Supervisor

TIME_STEP = 16

def load_position(file_path):
    try:
        with open(file_path, 'r') as file:
            line = file.readline().strip()
            x, y, z = map(float, line.split(','))
            return [x, y, z]
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return [0.0, 0.0, 0.0]  # Default value in case of error

def load_rotation(file_path):
    try:
        with open(file_path, 'r') as file:
            line = file.readline().strip()
            axis_x, axis_y, axis_z, angle = map(float, line.split(','))
            return [axis_x, axis_y, axis_z, angle]
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return [1.0, 0.0, 0.0, 0.0]  # Default value in case of error

def run_supervisor():
    robot = Supervisor()
    e_puck = robot.getFromDef('my_epuck')

    # Load position and rotation from files
    start_position = load_position(r'C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\test_circuit\start_position.txt')
    #start_rotation = load_rotation(r'C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\test_circuit\start_rotation.txt')

    # Set the translation and rotation fields
    translation_field = e_puck.getField('translation')
    #rotation_field = e_puck.getField('rotation')

    translation_field.setSFVec3f(start_position)
    #rotation_field.setSFRotation(start_rotation)

    # Run the simulation for one step
    i = 0
    while robot.step(TIME_STEP) != -1 and i < 1:
        i += 1

if __name__ == "__main__":
    run_supervisor()
