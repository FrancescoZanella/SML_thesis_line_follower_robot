@echo off
python "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\src\src\train_streaming_models.py" ^
    -output_dir "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\data\models" ^
    -dataset_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\data\\sensors_data\\sampled_umap_20_sensor_data_2024-08-23_18-01-07.csv" ^
    -model_name "leveraging_bagging"