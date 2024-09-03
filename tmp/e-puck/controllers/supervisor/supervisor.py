from controller import Supervisor

TIME_STEP = 32
DRIFT_FILE = r'C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\tmp\e-puck\controllers\controller_with_camera\drift_status.txt'

def main():
    robot = Supervisor()
    transform_node = robot.getFromDef('sign')
    shape_node = transform_node.getField('children').getMFNode(0)
    appearance_node = shape_node.getField('appearance').getSFNode()
    texture_node = appearance_node.getField('texture').getSFNode()
    
    url_field = texture_node.getField('url')
    
    drift_active = True
    
    while robot.step(TIME_STEP) != -1:
        # Leggi lo stato del drift dal file
        with open(DRIFT_FILE, 'r') as f:
            drift_status = f.read().strip()
        if drift_status == '1' and not drift_active:
            url_field.setMFString(0, 'C:\\Users\\franc\\Desktop\\drift.png')
            drift_active = True
            print('Drift attivo')
        elif drift_status == '0' and drift_active:
            url_field.setMFString(0, 'C:\\Users\\franc\\Desktop\\no_drift.png')
            drift_active = False
            print('Drift spento')

    
if __name__ == "__main__":
    main()
