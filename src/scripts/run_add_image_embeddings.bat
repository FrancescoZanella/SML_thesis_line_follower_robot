@echo off
python "C:\Users\franc\Desktop\TESI\SML_thesis_line_follower_robot\src\src\create_embeddings.py" ^
    -output_dir "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\data\\sensors_data\\umap_5_sensor_data 2024-08-21_16-06-41.csv" ^
    -dataset_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\data\\sensors_data\\sensor_data 2024-08-21_16-06-41.csv" ^
    -images_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\images\\" ^
    -model_path "C:\\Users\\franc\\Desktop\\TESI\\SML_thesis_line_follower_robot\\webots\\data\\models\\generate_embeddings_images.keras" ^
    -dimensionality_reduction "True" ^
    -n_components "5"