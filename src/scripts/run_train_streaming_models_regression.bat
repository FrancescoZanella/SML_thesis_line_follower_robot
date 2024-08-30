@echo off
python "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\src\src\train_streaming_models_regression.py" ^
    -output_dir "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\tmp\\e-puck\\data\\models" ^
    -dataset_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\tmp\\e-puck\\data\\data\\sensors_data\\sensors_data.csv" ^
    -model_name "arf"