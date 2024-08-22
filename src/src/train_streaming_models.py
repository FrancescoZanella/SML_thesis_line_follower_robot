import pandas as pd
import logging
import argparse
import datetime
from sklearn.model_selection import train_test_split
from pathlib import Path
from river import tree, metrics, forest, neighbors,ensemble, naive_bayes, stream, drift
from river.evaluate import progressive_val_score
from river.stream import iter_sklearn_dataset
import pickle
import re




LOGGING_FORMATTER = "%(asctime)s:%(name)s:%(levelname)s: %(message)s"


def main(dataset_path, output_dir,model_name,images):
    
    
    logging.info("Starting to load the dataset")
    logging.info(f"Dataset path: {dataset_path}")

    if images == 'True':
        df = pd.read_csv(dataset_path)
    elif images == 'False':
        df = pd.read_csv(dataset_path,names=['sensor0','sensor1','sensor2','sensor3','sensor4','sensor5','sensor6','target'])

    column_names = df.drop('target',axis=1).columns

    logging.info(f"Number of columns: {len(df.columns)}")
    models = {
        'naive_bayes': naive_bayes.GaussianNB(),
        'knn': neighbors.KNNClassifier(n_neighbors=10),
        'ht': tree.HoeffdingTreeClassifier(),
        'hat': tree.HoeffdingAdaptiveTreeClassifier(),
        'bagging': ensemble.BaggingClassifier(model=tree.HoeffdingTreeClassifier()),
        'leveraging_bagging': ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingTreeClassifier()),
        'arf': forest.ARFClassifier(),
        'srp': ensemble.SRPClassifier(model=tree.HoeffdingTreeClassifier(),
                      n_models=10,
                      drift_detector=drift.ADWIN(delta=0.001),
                      warning_detector=drift.ADWIN(delta=0.01),
                      seed=42),
        'adwin_bagging': ensemble.ADWINBaggingClassifier(model=tree.HoeffdingTreeClassifier())
    }
    model = models[model_name]
    metric = metrics.Accuracy()
    streams = stream.iter_pandas(X=df[column_names], y=df['target'])
    logging.info(f'TRAINING MODEL {model_name}')
    progressive_val_score(dataset=streams, 
                      model=model, 
                      metric=metric, 
                      print_every=1000)

      
    logging.info(f'Accuracy: {metric}')
    file_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    acc = re.findall(r"\d+.\d+", str(metric))[0] 
    out = output_dir.joinpath(f'{acc}_{model_name}_{file_name}.pkl')
    with open(out, 'wb') as f:
        pickle.dump(model, f)

    logging.info(f'Saved model at: {output_dir}')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for streaming models")
    parser.add_argument("-output_dir", default=None, type=str,
                        help="The directory where to save the model")
    parser.add_argument("-dataset_path", default=None, type=str, required=True,
                        help="Directory where the dataset is placed")
    parser.add_argument("-model_name", default=None, type=str, required=True, choices=['naive_bayes','knn','ht', 'hat', 'bagging', 'leveraging_bagging', 'arf','adwin_bagging','srp' ],
                        help="model type to be trained")
    parser.add_argument("-with_images", default=None, type=str, required=True,
                        help="True if the input dataset has also images informations")
    
    args = parser.parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    DATASET_DIR = Path(args.dataset_path)
    MODEL_NAME = args.model_name
    IMAGES = args.with_images
    
   
    
    
    
    
     
    
    log_path = OUTPUT_DIR / "log.txt"
    logging.basicConfig(format=LOGGING_FORMATTER, level=logging.INFO, force=True)
    
    
    
    main(DATASET_DIR, OUTPUT_DIR,MODEL_NAME,IMAGES)
