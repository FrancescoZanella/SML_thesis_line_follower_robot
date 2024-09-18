# SML Thesis Line Follower Robot

This repository contains the code and resources for a line-following robot using streaming machine learning techniques, the robot thanks to incremental learnign and catastrophic forgetting avoidance techniques is able to follow the line even when concept drift appear. 

The robot can operate in two modes:
- Training mode, where it collects data used to train models
- Production mode: the robot uses a pre-trained model built with River to navigate, continuously updating the model as the robot moves adapting to the drift.

## Getting Started
### Prerequisites

1. **Webots**: Download and install Webots from [this page](https://cyberbotics.com/).
2. **Git**: Make sure you have Git installed on your system.
3. **Python**: Python >=3.11 is required.

### Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/FrancescoZanella/SML_thesis_line_follower_robot.git

    cd SML_thesis_line_follower_robot
    ```

2. **Create a virtual environment and install dependencies**:

    ```bash
    python -m venv env_name
    ```

    - For macOS/Linux:
    
        ```bash
        source env_name/bin/activate
        ```

    - For Windows:
    
        ```bash
        env_name\Scripts\activate
        ```

    - Install required packages:

        ```bash
        pip install -r requirements.txt
        ```

### Setup in Webots

1. **Open and configure Webots**:
    - Go to `File -> Open World`.
    - Navigate to `../SML_thesis_line_follower_robot/e-puck/worlds/e-puck_line.wbt` 
    - Then go to `Tools -> Preferences` and set python command to:

      On Windows:
        - `…\SML_thesis_line_follower_robot\tesi\Scripts\python.exe`

      On MacOs/Linux:
        - `…/SML_thesis_line_follower_robot/tesi/bin/python`

      **N.B.** *write the full path to python executable inside the venv just created.*

2. **Configure the Robot Controller**:
    - In the left dropdown menu, select the `epuck` node.
    - Modify the `controllerArgs` field to specify the following parameters:

        - **PRODUCTION**: Determines whether the robot operates in production mode (`True`) or in data collection mode (`False`). In production mode, the robot uses a pre-trained model to make decisions, while in data collection mode, it gathers data to train or improve a model.

        - **MODEL_PATH**: Specifies the file path to the pre-trained model that the robot will load and use for making predictions. This path is crucial when the robot is running in production mode.

        - **PLOT**: If set to `True`, the program generates and saves plots in `plots/` folder of the model's accuracy over time during the robot’s operation with drift or without drift. This visual representation helps in evaluating the model's performance as the robot executes its tasks.

        - **SAVE_SENSORS**: When `True`, this parameter instructs the robot to save the sensor data it collects during the run. This data is then used to train a model or improve an existing one.

        - **SAVE_IMAGES**: If set to `True`, the program saves the images captured by the robot's camera throughout its operation. These images can then be used to build embeddings or for future usage.

        - **VERBOSE**: When `True`, the robot provides detailed output during its operation, including predicted labels, actual labels, and the model’s accuracy at each step. This is helpful for debugging and understanding how the robot is performing its tasks in real-time.

        - **LEARNING**: Determines whether the model should operate in online learning mode during inference. If set to `True`, the model will continue to learn and adapt in the presence of drift, allowing it to update its knowledge base in real-time. If set to `False`, the model will behave like a traditional batch machine learning model, maintaining its initial training without further updates.

        - **ENABLE_RECOVERY**: When set to `True`, this parameter activates an additional algorithm that utilizes the robot's recent velocity data. If the robot loses track of the line, this feature enables it to attempt recovery by using the stored velocity information to guide its movements. This can be particularly useful in maintaining the robot's intended path even in challenging situations where the line may be temporarily lost.

## Operating Modes

The robot can operate in two different modes:

### 1. Training Mode

- **PRODUCTION = False**: The robot employs a controller that monitors the difference between the values recorded on the right and left sides of the line. This controller enables the robot to navigate while simultaneously collecting valuable data. The collected data, including sensor readings and images (when *SAVE_SENSORS = True* and *SAVE_IMAGES = True*), is crucial for training the machine learning model that will be used in production mode.

**Steps to run in Training Mode and collect data**:

• Run the robot in Webots with `PRODUCTION=False`, `SAVE_SENSORS=True`, and `SAVE_IMAGES=True` to collect data.

• Once you have collected enough data, stop the simulation.

• The collected data will be stored in the `data\sensor_data` with the name of the current date and time. This structure ensures easy access and management of your training datasets.

• Run `.\SML_thesis_line_follower_robot\src\scripts\run_train_streaming_models_regression` script(.sh for Mac/Linux .bat for Windows), passing the CSV file with the collected data and the type of model as input. This script will train the model using the specified parameters and save the model in `models` folder.


### 2. Production Mode

- **PRODUCTION = True**: The robot utilizes a combination of a pre-trained model (built with data collected during training) and incremental learning. It navigates based on the pre-trained model while continuously adapting its model in response to detected drift. This approach allows the robot to maintain optimal performance by leveraging both its initial training and real-time learning from new environmental conditions.

**Steps to run in Production Mode**:
1. Prepare the model:

   a) If you already have a trained model:
      - Ensure the model is saved in the `models` folder.
      - Verify that the `MODEL_PATH` in the controller arguments in the webots interface points to your existing model.

   b) If you want to create a new model from scratch:
      - Run the robot in Training Mode (PRODUCTION=False) to collect data.
      - Use the collected data to train a new model.
      - Save the newly trained model in the `models` folder.
      - Update the `MODEL_PATH` in the controller arguments to point to your new model.
2. Run the robot in Webots with `PRODUCTION=True`.
3. Set the `LEARNING=False` if you want to deactivate the incremental learning.
4. Monitor and analyze performance:
   - If `PLOT=True`, the program will generate and save plots in the `plots/` folder.
   - These plots visualize the model's accuracy over time, both with and without drift.
   - Review these plots to evaluate the model's performance and the effectiveness of incremental learning.





## Drift Intervals

The current simulation includes predefined drift intervals to test the robot's ability to adapt to changing conditions. These intervals are:

- First drift interval: 500-3000 steps
- Second drift interval: 3500-4500 steps

During these intervals, the envirnoment change from a day setup to a night setup, challenging the robot's line-following capabilities.

### Customizing Drift Intervals

You can easily modify these drift intervals to introduce concept drift at different times during the simulation. To do this:

1. In the Webots interface, locate the `supervisor` node in the scene tree.
2. Expand the `supervisor` node and find the `controllerArgs` field.
3. Modify the values in this field to set your desired drift intervals.

