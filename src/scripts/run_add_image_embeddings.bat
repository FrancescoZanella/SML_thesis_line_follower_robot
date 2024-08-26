@echo off
python "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\src\src\create_embeddings.py" ^
    -output_dir "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\data\\sensors_data\\umap_20_sensor_data_2024-08-23_18-01-07.csv" ^
    -dataset_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\data\\sensors_data\\sensor_data_2024-08-23_18-01-07.csv" ^
    -images_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\images\\" ^
    -model_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\models\\image_model\\generate_embeddings_images.keras" ^
    -dimensionality_reduction "True" ^
    -n_components "20"