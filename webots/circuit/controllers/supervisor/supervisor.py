from controller import Supervisor
import random



# function to generate a random color
def generate_random_color():
    value = random.randint(0, 1)
    return [value, value, value]

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
  if j % 10 == 0:
    #print(f'enter_{j}')
    base_color_field.setSFColor([rgb/255,rgb/255,rgb/255])
    #print(rgb/255,rgb/255,rgb/255)
    if rgb < 180:
      down = False
    if rgb > 254:
      down = True

    if down:
      rgb=rgb-1
    else:
      rgb=rgb+1
  else:
    #print(j)
    #print(rgb/255,rgb/255,rgb/255)
    pass
  
  j+=1
  
  



  