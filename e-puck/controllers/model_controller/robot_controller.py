from controller import Robot
import csv
from datetime import datetime
import math
from pathlib import Path
import pickle
from river import metrics
from matplotlib import pyplot as plt
from drift_detector import DualDriftDetector

class RobotController:
    TIME_STEP = 32
    MAX_SPEED = 6.28
    LFM_FORWARD_SPEED = 200
    NB_GROUND_SENS = 3
    

    def __init__(self, production, model_path, plot, save_sensors, verbose, learning):
        self.production = production
        self.model_path = model_path
        self.plot = plot
        self.save_sensors = save_sensors
        self.verbose = verbose
        self.learning = learning

        self.mem = 0
        self.MAX_MEM = 2000
        self.decay_rate = -math.log(0.001) / self.MAX_MEM

        self.robot = Robot()
        self.sensors,self.left_motor, self.right_motor = self.initialize_devices()
        self.create_directories()

        if self.production:
            self.pretrained_model, self.pretrained_metric = self.load_model()

        self.drift_detector = DualDriftDetector(300, 850, 50, 20, 25)
        self.mae_log = []
        self.labels = []
        self.sensors_data = []

    def initialize_devices(self):
        left_motor = self.robot.getDevice('left wheel motor')
        right_motor = self.robot.getDevice('right wheel motor')
        left_motor.setPosition(float('inf'))
        right_motor.setPosition(float('inf'))
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        camera = self.robot.getDevice('camera')
        camera.enable(self.TIME_STEP)

        sensors = []
        for i in range(self.NB_GROUND_SENS):
            sensor = self.robot.getDevice(f'gs{i}')
            sensor.enable(self.TIME_STEP)
            sensors.append(sensor)

        return sensors,left_motor, right_motor

    def load_model(self):
        with open(f'{self.model_path}', 'rb') as f:
            pretrained_model = pickle.load(f)
        
        return pretrained_model, metrics.MAE()

    def create_directories(self):
        base_path = Path(self.model_path).parent.parent
        data_path = base_path.joinpath('data')
        sensors_data_path = data_path.joinpath('sensors_data')
        plots_path = base_path.joinpath('plots')
        
        data_path.mkdir(parents=True, exist_ok=True)
        sensors_data_path.mkdir(parents=True, exist_ok=True)
        plots_path.mkdir(parents=True, exist_ok=True)
        

    def get_sensors_data(self):
        sensor_data_t = []
        for i in range(self.NB_GROUND_SENS):
            sensor_data_t.append(self.sensors[i].getValue())

        s_dict = {f'sensor{j}': val for j, val in enumerate(sensor_data_t)}
        
        return s_dict, sensor_data_t
    

    def load_true_labels(self, irs_values,sensible):
        DeltaS = irs_values[2] - irs_values[0]
        if sensible:
            speed_l = self.LFM_FORWARD_SPEED - 1e-6 * math.pow(DeltaS, 3)
        else:
            speed_l = self.LFM_FORWARD_SPEED - 0.6 * math.pow(DeltaS, 1)
        return speed_l

    def run(self):
        i = 0
        last_is_drift = False

        while self.robot.step(self.TIME_STEP) != -1:
            X, irs_values = self.get_sensors_data()

            self.sensors_data.append(irs_values)

            self.drift_detector.update(irs_values[0],irs_values[2])
            
            if self.production:
                vel,last_is_drift = self.handle_production_mode(X, irs_values, last_is_drift)
            else:
                vel = self.load_true_labels(irs_values,sensible=True)
                self.labels.append(vel)

            self.update_motor_speeds(vel, i)
            i += 1

        self.post_run_actions()

    def handle_production_mode(self, X, irs_values, last_is_drift):

        is_drift = self.drift_detector.drift_detected
        
        # PREDICT
        if is_drift and self.learning:

            # continuos drift
            if last_is_drift:
                vel_pred = self.model.predict_one(X)
            
            # create and use new model from scratch
            else:
                print('Creating a new model from scratch')
                self.model,self.metric = self.load_model()
                self.mem = 0
                vel_pred = self.model.predict_one(X)
            vel_true = self.load_true_labels(irs_values, sensible=False)
            self.metric.update(vel_true, vel_pred)
        else:
            if last_is_drift and self.learning:
                print(f'Deleting model at step')
                del self.model
                del self.metric
                self.mem = 0

            vel_pred = self.pretrained_model.predict_one(X)
            vel_true = self.load_true_labels(irs_values, sensible=True)
            self.pretrained_metric.update(vel_true,vel_pred)

        # UPDATE
        if is_drift and self.learning:
            self.mae_log.append(self.metric.get())
            if self.verbose:
                print(f'MAE: {self.metric.get()}')
        else:
            self.mae_log.append(self.pretrained_metric.get())
            if self.verbose:
                print(f'MAE: {self.pretrained_metric.get()}')

        self.labels.append(vel_true)
        
        


        # LEARN
        if self.learning:
            if is_drift:
                if self.mem <= self.MAX_MEM:
                    weight = 3 * math.exp(-self.decay_rate * self.mem)
                    self.model.learn_one(X, vel_true,sample_weight=weight)
                    self.mem += 1
                else:
                    print('Stop learning, exceding')

        
        
        return vel_pred,is_drift


    def update_motor_speeds(self, vel, i):

        
        left_speed = max(min(0.00628 * vel, self.MAX_SPEED), -self.MAX_SPEED)
        right_speed = max(min(0.00628 * (400 - vel), self.MAX_SPEED), -self.MAX_SPEED)


        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)


    def post_run_actions(self):

        if self.plot and self.production:
            
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            study_dir = f"study_{current_time}"
            plots_path = Path.cwd().parent.parent.joinpath('data','plots', str(study_dir))
            plots_path.mkdir(parents=True, exist_ok=True)
            
            self.plot_MAE_drift(plots_path)
        if self.save_sensors and not self.production:
            self.save_sensor_data()


    def save_sensor_data(self):
        sensors_data_path = Path(self.model_path).parent.parent.joinpath('data', 'sensors_data', f'sensors_data_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
        sensors_data_path.parent.mkdir(parents=True, exist_ok=True)

        with open(sensors_data_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['sensor0', 'sensor1', 'sensor2', 'target'])

            for irs, vel in zip(self.sensors_data, self.labels):
                csv_writer.writerow(irs + [vel])

    def plot_MAE_drift(self,path):

        plt.figure(figsize=(12, 6))
        plt.plot(self.mae_log, label='MAE', zorder=10)
        plt.title('MAE over time with Drift Detection')
        plt.xlabel('Time steps')
        plt.ylabel('MAE')
        
        for start, end in self.drift_detector.anomalies:
            plt.axvspan(start, end, facecolor='red', alpha=0.2, label='Concept')
            plt.axvline(x=start, color='red', linestyle='--', linewidth=1, label='Concept Drift')
            plt.axvline(x=end, color='red', linestyle='--', linewidth=1, label='Concept Drift')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')
        plt.savefig(path.joinpath('MAE_drift.png'))