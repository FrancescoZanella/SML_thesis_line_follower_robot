
# SML Thesis Line Follower Robot

This repository contains the code and resources for a line-following robot using streaming machine learning techniques. 

The robot can operate in two modes: 
- Training mode, where it collects data used to train models
- Production mode: the robot uses a pre-trained model built with River to navigate, continuously updating the model as the robot moves.

## Getting Started

### Prerequisites

1. **Webots**: Download and install Webots from [this page](https://cyberbotics.com/).
2. **Git**: Make sure you have Git installed on your system.
3. **Python**: Python 3.11 is required.

### Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/FrancescoZanella/SML_thesis_line_follower_robot.git
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

1. **Open Webots**:
    - Go to `File -> Open World`.
    - Navigate to `webots/circuit/worlds/circuit.wbt` and open it.

2. **Configure the Robot Controller**:
    - In the left dropdown menu, select the `epuck` node.
    - Modify the `controllerArgs` field to specify the following parameters:

        - **PRODUCTION**: Determines whether the robot operates in production mode (`True`) or in data collection mode (`False`). In production mode, the robot uses a pre-trained model to make decisions, while in data collection mode, it gathers data to train or improve a model.

        - **MODEL_PATH**: Specifies the file path to the pre-trained model that the robot will load and use for making predictions. This path is crucial when the robot is running in production mode.

        - **PLOT**: If set to `True`, the program generates and saves a plot of the model's accuracy over time during the robot’s operation. This visual representation helps in evaluating the model's performance as the robot executes its tasks.

        - **SAVE_SENSORS**: When `True`, this parameter instructs the robot to save the sensor data it collects during the run. This data is then used to train a model or improve an existing one.

        - **SAVE_IMAGES**: If set to `True`, the program saves the images captured by the robot's camera throughout its operation. These images are crucial as they are processed by a pre-trained Convolutional Neural Network (CNN) to generate features that are used for training or improving the streaming machine learning model.

        - **VERBOSE**: When `True`, the robot provides detailed output during its operation, including predicted labels, actual labels, and the model’s accuracy at each step. This is helpful for debugging and understanding how the robot is performing its tasks in real-time.

## Operating Modes

The robot can operate in two different modes:

### 1. Training Mode

- **PRODUCTION = False**: The robot uses a PID controller to navigate and collects data for training a model, *SAVE_SENSORS = True* and *SAVE_IMAGES = True* to collect sensors data and images used in future training.

**Steps to run in Training Mode**:
1. Run the robot in Webots with `PRODUCTION=False`, `SAVE_SENSORS=True`, and `SAVE_IMAGES=True` to collect data.
2. Once you have collected enough data, stop the simulation.
3. Generate embeddings:

    - Activate the virtual environment:

        ```bash
        source env_name/bin/activate  # For macOS/Linux
        env_name\Scripts\activate  # For Windows
        ```

    - Run the script to add image embeddings:

        ```bash
        python run_add_image_embeddings.py  # Generates UMAP and full embeddings
        ```

4. Train the streaming model using the collected data:

    ```bash
    python run_train_streaming_model.py
    ```

### 2. Production Mode

- **PRODUCTION = True**: The robot uses the pre-trained model to navigate based on the collected features and image embeddings.

**Steps to run in Production Mode**:
1. Ensure the model is trained and saved at the specified `MODEL_PATH`.
2. Run the robot in Webots with `PRODUCTION=True`.


