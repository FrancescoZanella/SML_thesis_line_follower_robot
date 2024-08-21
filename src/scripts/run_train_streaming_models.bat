@echo off
python "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\src\src\train_streaming_models.py" ^
    -output_dir "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\data\models" ^
    -dataset_path "C:\\Users\\franc\\Desktop\\TRAINING_DATA\\df_tot_only_sensors.csv" ^
    -model_name "arf" ^
    -evaluate "True"