@echo off
python "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\src\src\create_embeddings.py" ^
    -output_dir "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\tmp\\e-puck\\data\\data\\sensors_data\\sensor_with_emb_5.csv" ^
    -dataset_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\tmp\\e-puck\\data\\data\\sensors_data\\sensors_data.csv" ^
    -images_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\tmp\\e-puck\\data\\images" ^
    -model_path "C:\\Users\\franc\\Desktop\\prova.keras" ^
    -dimensionality_reduction "True" ^
    -n_components "5"