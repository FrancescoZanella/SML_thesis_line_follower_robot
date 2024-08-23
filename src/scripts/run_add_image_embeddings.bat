@echo off
python "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\src\src\create_embeddings.py" ^
    -output_dir "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\data\\sensors_data\\full_training_data.csv" ^
    -dataset_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\data\\sensors_data\\sensor_data_2024-08-23_17-52-28.csv" ^
    -images_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\images\\" ^
    -model_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\models\\image_model\\generate_embeddings_images.keras" ^
    -dimensionality_reduction "False" ^
    -n_components "10"