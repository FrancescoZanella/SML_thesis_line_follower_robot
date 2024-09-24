import pandas as pd
import logging
import argparse
import datetime
from sklearn.model_selection import train_test_split
from pathlib import Path
from river import tree, metrics, forest, neighbors,ensemble, naive_bayes, stream, drift
from river.evaluate import progressive_val_score
import pickle
import re
from river import linear_model, rules, preprocessing
from Oespl import OESPL
LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(dataset_path, output_dir,model_name):
    
    
    logging.info("Starting to load the dataset")
    logging.info(f"Dataset path: {dataset_path}")

    df = pd.read_csv(dataset_path)
    
    
    column_names = df.drop('target',axis=1).columns
    X = df.drop('target',axis=1)
    y = df['target']


    logging.info(f"Number of columns: {len(df.columns)}")
    if len(df.columns) == 9:
        name = 'umap_5'
    if len(df.columns) == 24:
        name = 'umap_20'
    if len(df.columns) == 14:
        name = 'umap_10'
    if len(df.columns) == 220:
        name = 'full_emb'
    if len(df.columns)==4:
        name = 'raw'
    models = {
        'linear_regression': linear_model.LinearRegression(),
        'knn': neighbors.KNNRegressor(n_neighbors=1000),
        'amrules': rules.AMRules(
            delta=0.01,
            n_min=100,
            drift_detector=drift.ADWIN(),
            pred_type='adaptive'
        ),
        'ht': tree.HoeffdingTreeRegressor(
            grace_period=100,
            leaf_prediction='adaptive',
            model_selector_decay=0.9
        ),
        'hat': tree.HoeffdingAdaptiveTreeRegressor(),
        'arf': forest.ARFRegressor(
            n_models=10,
            seed=1,
            model_selector_decay=0.9,           
            leaf_prediction='adaptive'
        ),
        'srp': ensemble.SRPRegressor(
            n_models=10,
            seed=1,
            drift_detector=drift.ADWIN(delta=0.001),
            warning_detector=drift.ADWIN(delta=0.01),
            model = tree.HoeffdingTreeRegressor(
                grace_period=100,
                leaf_prediction='adaptive',
                model_selector_decay=0.9,                
            )
        ),
        'oespl': OESPL(
            base_estimator=tree.HoeffdingTreeRegressor(max_depth=32),
            ensemble_size=10,
            lambda_fixed=6.0,
            seed=42,
            drift_detector=drift.ADWIN(),
            patience=1000,
            awakening=500,
            reset_model=True
        )
        
    }
    
    model = (preprocessing.StandardScaler() | models[model_name])
    metric = metrics.MAE()

    logging.info(f'TRAINING MODEL {model_name}')
    

   
    streams1 = stream.iter_pandas(X, y, shuffle=True, seed=42)
    result1 = progressive_val_score(dataset=streams1, model=model, metric=metric, print_every=1000)
    print(f'Progressive validation score: {result1}')

    
    

    file_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    acc = re.findall(r"\d+.\d+", str(metric))[0] 
    out = output_dir.joinpath(f'{acc}_{name}_{model_name}_{file_name}.pkl')
    with open(out, 'wb') as f:
        pickle.dump(model, f)

    logging.info(f'Saved model at: {output_dir}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for streaming models")
    parser.add_argument("-output_dir", default=None, type=str,
                        help="The directory where to save the model")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-model_name", default=None, type=str, required=True, choices=['linear_regression','knn','ht', 'hat', 'arf', 'amrules','srp','oespl' ],
                        help="model type to be trained")

    
    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    DATASET_DIR = Path(args.dataset_path)
    MODEL_NAME = args.model_name
    
    
   
    
    
    
    
     
    
    log_path = OUTPUT_DIR / "log.txt"
    logging.basicConfig(format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    
    
    main(DATASET_DIR, OUTPUT_DIR,MODEL_NAME)
