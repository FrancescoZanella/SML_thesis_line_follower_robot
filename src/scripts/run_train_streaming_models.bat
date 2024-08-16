@echo off
python "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\src\src\train_streaming_models.py" ^
    -output_dir "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\data\models" ^
    -dataset_path "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\data\data\sensors_data\sensor_data 2024-08-16_14-38-58.csv" ^
    -model_name "hat" ^
    -evaluate "True"