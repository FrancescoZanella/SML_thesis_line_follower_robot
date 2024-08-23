@echo off
python "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\src\src\train_streaming_models.py" ^
    -output_dir "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\webots\data\models" ^
    -dataset_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\data\\sensors_data\\umap_10_sensor_data 2024-08-21_16-06-41.csv" ^
    -model_name "arf"