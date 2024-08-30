from controller import Supervisor
import random



TIME_STEP = 16

robot = Supervisor()  # create Supervisor instance


floor = robot.getFromDef('floor')


floorAppearance = floor.getField('floorAppearance')


appearance_node = floorAppearance.getSFNode()
base_color_field = appearance_node.getField('baseColor')


j = 0
rgb = 255
down = True
while robot.step(TIME_STEP) != -1:
  
  if j % 500 == 0:
    print(f'Change_at_{j}')
    base_color_field.setSFColor([rgb/255,rgb/255,rgb/255])
    if rgb < 100:
      down = False
    if rgb > 254:
      down = True

    if down:
      rgb=rgb-60
    else:
      rgb=rgb+60
  
  j+=1
  
  



  