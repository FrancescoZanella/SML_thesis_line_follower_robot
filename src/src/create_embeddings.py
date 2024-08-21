import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras import layers as tfkl
from tensorflow import keras as tfk
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import re
import logging
import argparse
import datetime
from pathlib import Path




LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(dataset_path, output_dir, image_folder,model_path):
    
    
    logging.info("Starting to load the dataset")
    logging.info(f"Dataset path: {dataset_path}")

    
    df = pd.read_csv(dataset_path,names=['sensor0','sensor1','sensor2','sensor3','sensor4','sensor5','sensor6','target'])

    df.reset_index(inplace=True)
    
    logging.info(f"Model path: {model_path}")

    model = tfk.models.load_model(model_path)

    embeddings_list = []
    indices = []
    i = 0
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path)
  
        image = tf.cast(image, tf.float32) / 255.0
    
        flatten_layer = tf.keras.Sequential(model.layers[:8])
    
        embedding = flatten_layer(tf.expand_dims(image, axis=0))
        
        embedding = tf.squeeze(embedding, axis=0).numpy()
    
        index = int(re.findall(r"\d+", img_name)[0])
        
        indices.append(index)
        embeddings_list.append(embedding)
        print(f'{i}/{len(df)}')
        i+=1

    
    df_emb = pd.DataFrame(embeddings_list, columns=[f'embedding_{i+1}' for i in range(216)])
    df_emb.insert(0, 'index', indices)
    df_emb = df_emb.sort_values(by='index')

    df_tot = pd.merge(df, df_emb, on='index', how='inner').drop('index',axis=1)

    df_tot.to_csv(output_dir,index=False)
    
    logging.info(f'Saved model at: {output_dir}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for streaming models")
    parser.add_argument("-output_dir", default=None, type=str,
                        help="The directory where to save the model")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-images_path", default=None, type=str, required=True,
                        help="Directory where the images to use to generate embeddings are")
    parser.add_argument("-model_path", default=None, type=str, required=True,
                        help="model to use to create embeddings")
    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    DATASET_DIR = args.dataset_path
    IMAGES_PATH = args.images_path
    MODEL_PATH = args.model_path

    
    
     
    
    log_path = OUTPUT_DIR / "log.txt"
    logging.basicConfig(format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    
    
    main(DATASET_DIR, OUTPUT_DIR,IMAGES_PATH,MODEL_PATH)
