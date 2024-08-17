from controller import Robot, Camera
import random

# Inizializza il robot
robot = Robot()

# Definisci il time step
TIME_STEP = 64

# Inizializza i motori
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Inizializza la telecamera
camera = robot.getDevice('camera')
camera.enable(TIME_STEP)

# Funzione per impostare velocità casuali
def set_random_speed():
    left_speed = random.uniform(-6.28, 6.28)
    right_speed = random.uniform(-6.28, 6.28)
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

# Ciclo principale
while robot.step(TIME_STEP) != -1:
    # Imposta velocità casuali ai motori
    set_random_speed()

    # La telecamera è già attiva e visualizzabile nel riquadro di Webots
    # Non è necessario fare altro per visualizzare l'immagine

    # Puoi opzionalmente aggiungere un ritardo o una logica per cambiare direzione dopo un certo tempo

# Cleanup non necessario in Webots
