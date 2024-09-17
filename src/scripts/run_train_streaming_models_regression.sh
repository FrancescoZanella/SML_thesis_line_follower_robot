#!/bin/bash
python ~/Desktop/TESI/SML_thesis_line_follower_robot/src/src/train_streaming_models_regression.py \
    -output_dir '~/Desktop/TESI/SML_thesis_line_follower_robot/tmp/e-puck/data/models' \
    -dataset_path '~/Desktop/TESI/SML_thesis_line_follower_robot/tmp/e-puck/data/data/sensors_data/sensor_with_emb_5.csv' \
    -model_name 'arf'