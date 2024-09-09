from controller import Supervisor
from pathlib import Path

TIME_STEP = 32
DRIFT_FILE = Path(__file__).parent.parent / 'controller_with_camera' / 'drift_status.txt'
STATUS_FILE = Path(__file__).parent.parent.parent / 'resources'
def main():
    robot = Supervisor()
    transform_node = robot.getFromDef('sign')
    shape_node = transform_node.getField('children').getMFNode(0)
    appearance_node = shape_node.getField('appearance').getSFNode()
    texture_node = appearance_node.getField('texture').getSFNode()
    
    floor = robot.getFromDef('floor')
    floorAppearance = floor.getField('floorAppearance')
    appearance_node = floorAppearance.getSFNode()
    base_color_field = appearance_node.getField('baseColor')
    
    url_field = texture_node.getField('url')

    drift_intervals = [(1000,5000),(7000,10000)]
    
    drift_active = False
    
    i = 0
    base_color_field.setSFColor([255/255,255/255,255/255])
    url_field.setMFString(0, str(STATUS_FILE / 'no_drift.png'))
    while robot.step(TIME_STEP) != -1:
        if any(start <= i < end for start, end in drift_intervals) and not drift_active:
            base_color_field.setSFColor([100/255,100/255,100/255])
            url_field.setMFString(0, str(STATUS_FILE / 'drift.png'))
            drift_active = True
            print(f'{i} Drift attivo')
        if not any(start <= i < end for start, end in drift_intervals) and drift_active:
            url_field.setMFString(0, str(STATUS_FILE / 'no_drift.png'))
            drift_active = False
            base_color_field.setSFColor([255/255,255/255,255/255])
            print(f'{i} Drift spento')
        i+=1

    
if __name__ == "__main__":
    main()






  



  