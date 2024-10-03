from controller import Supervisor
from pathlib import Path
import sys
import os

TIME_STEP = 32
STATUS_FILE = Path(__file__).parent.parent.parent / 'resources'
def main():
    robot = Supervisor()

    transform_node = robot.getFromDef('sign')
    shape_node = transform_node.getField('children').getMFNode(0)
    appearance_node = shape_node.getField('appearance').getSFNode()
    texture_node = appearance_node.getField('texture').getSFNode()
    url_field = texture_node.getField('url')

    floor = robot.getFromDef('floor')
    floorAppearance = floor.getField('floorAppearance')
    appearance_node = floorAppearance.getSFNode()
    base_color_field = appearance_node.getField('baseColor')
    
    

    button = robot.getDevice("button")
    button.enable(TIME_STEP)
    

    
    drift_active = False
    button_pressed = False
    
    i = 0
    base_color_field.setSFColor([255/255,255/255,255/255])
    url_field.setMFString(0, str(STATUS_FILE / 'DAY.png'))

    
    while robot.step(TIME_STEP) != -1:
        touch_value = button.getValue()
            
        
            
        if touch_value > 0:
            if not button_pressed:
                drift_active = not drift_active
                if drift_active:
                    base_color_field.setSFColor([100/255,100/255,100/255])
                    url_field.setMFString(0, str(STATUS_FILE / 'NIGHT.png'))
                else:
                    base_color_field.setSFColor([255/255,255/255,255/255])
                    url_field.setMFString(0, str(STATUS_FILE / 'DAY.png'))
                button_pressed = True
        else:
            button_pressed = False
        i+=1


if __name__ == "__main__":
    main()






  



  