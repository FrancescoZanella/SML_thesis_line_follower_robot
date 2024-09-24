@echo off
python "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\src\src\train_streaming_models_regression.py" ^
    -output_dir "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\e_puck\\data\\models" ^
    -dataset_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\e_puck\\data\\data\\sensors_data\\train_data.csv" ^
    -model_name "oespl"