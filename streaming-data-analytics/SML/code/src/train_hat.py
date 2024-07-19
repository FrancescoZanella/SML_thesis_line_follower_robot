import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import logging
import argparse
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from river import tree, preprocessing, metrics
from river.evaluate import progressive_val_score
from river.stream import iter_sklearn_dataset
import pickle









LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(dataset_path, output_dir, evaluate):
    
    
    logging.info("Starting to load the dataset")
    logging.info(f"Dataset path: {dataset_path}")

    column_names = [f'ir_{x}' for x in range(0,7)]
    column_names.append('target')
    df = pd.read_csv(dataset_path) 
    df.columns = column_names

        
    

    
    model = preprocessing.StandardScaler() | tree.HoeffdingAdaptiveTreeClassifier()
    metric = metrics.Accuracy()
    
    for _,x in df.iterrows():
        X = x[[f'ir_{j}' for j in range(0,7)]]
        y = x["target"]
        y_p = model.predict_one(X)   # Predict class
        if y_p is not None:
            metric.update(y_true=y, y_pred=y_p)
        model.learn_one(X, y)        # Train the model

      
    print(metric)
    
    with open(output_dir.joinpath('model.pkl'),'wb') as f:
        pickle.dump(model,f)

    logging.info(f'Saved model at: {output_dir}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for catboost")
    parser.add_argument("-output_dir", default=None, type=str,
                        help="The directory where to save the model")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-evaluate", default=False, type=str,
                        help="True if you wan to to evaluate the model")
    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    DATASET_DIR = Path(args.dataset_path)
    EVALUATE = args.evaluate
    
    
    
    
     
    
    log_path = os.path.join(OUTPUT_DIR, "log.txt")
    logging.basicConfig(filename=log_path, filemode="w", format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter(LOGGING_FORMATTER))

    root_logger = logging.getLogger()
    root_logger.addHandler(stdout_handler)
    
    main(DATASET_DIR, OUTPUT_DIR,EVALUATE)
