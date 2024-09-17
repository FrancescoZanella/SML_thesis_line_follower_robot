import numpy as np
from river import base, tree, drift, metrics, evaluate
from river.ensemble import AdaptiveRandomForestClassifier

class OESPL(base.Classifier):
    def __init__(self, base_estimator=tree.HoeffdingTreeClassifier(), 
                 ensemble_size=1, lambda_fixed=6.0, seed=1, 
                 drift_detector=drift.ADWIN(), patience=3000, 
                 awakening=1000, reset_model=False):
        self.base_estimator = base_estimator
        self.ensemble_size = ensemble_size
        self.lambda_fixed = lambda_fixed
        self.seed = seed
        self.drift_detector = drift_detector
        self.patience = patience
        self.awakening = awakening
        self.reset_model = reset_model
        
        self.ensemble = None
        self.random_state = np.random.RandomState(self.seed)
        self.n_instances = []
        self.drift_detectors = []
        self.mean_difference = []
        self.past_mean_difference = []
        self.awakening_counters = []
        self.patience_counters = []
        self.page_hinkley = []

    def _init_ensemble(self):
        self.ensemble = [base.clone(self.base_estimator) for _ in range(self.ensemble_size)]
        for i in range(self.ensemble_size):
            self.page_hinkley.append(drift.PageHinkley())
            self.mean_difference.append(True)
            self.past_mean_difference.append(None)
            self.awakening_counters.append(0)
            self.patience_counters.append(self.patience)
            self.n_instances.append(0)
            self.drift_detectors.append(base.clone(self.drift_detector))

    def learn_one(self, x, y):
        if self.ensemble is None:
            self._init_ensemble()

        for i in range(self.ensemble_size):
            self.n_instances[i] += 1
            
            # Page Hinkley test
            tree_size = len(self.ensemble[i]) if hasattr(self.ensemble[i], '__len__') else 0
            self.page_hinkley[i].update(tree_size)
            
            if self.n_instances[i] % self.patience_counters[i] == 0:
                change = self.page_hinkley[i].drift_detected
                self.past_mean_difference[i] = self.mean_difference[i]
                self.mean_difference[i] = change
                self.page_hinkley[i] = drift.PageHinkley()
                
                if not change and not self.past_mean_difference[i]:
                    self.awakening_counters[i] = self.awakening
                    self.patience_counters[i] *= 2
                elif change and not self.past_mean_difference[i]:
                    self.awakening_counters[i] = 0
                    self.patience_counters[i] = self.patience
            
            # Determine lambda
            if self.mean_difference[i] or self.awakening_counters[i] > 0:
                lambda_val = self.lambda_fixed
                if self.awakening_counters[i] > 0:
                    self.awakening_counters[i] -= 1
            else:
                lambda_val = 0.1
            
            # Poisson sampling
            k = self.random_state.poisson(lambda_val)
            if k > 0:
                for _ in range(k):
                    self.ensemble[i].learn_one(x, y)
            
            # Drift detection
            self._drift_detection(x, y, i)

        return self

    def _drift_detection(self, x, y, i):
        y_pred = self.ensemble[i].predict_one(x)
        correctly_classifies = int(y_pred == y)
        
        self.drift_detectors[i].update(1 - correctly_classifies)
        
        if self.drift_detectors[i].drift_detected:
            self.n_instances[i] = 0
            self.drift_detectors[i] = base.clone(self.drift_detector)
            self.mean_difference[i] = True
            self.past_mean_difference[i] = None
            self.page_hinkley[i] = drift.PageHinkley()
            self.awakening_counters[i] = 0
            self.patience_counters[i] = self.patience
            if self.reset_model:
                self.ensemble[i] = base.clone(self.base_estimator)

    def predict_proba_one(self, x):
        if self.ensemble is None:
            self._init_ensemble()
        
        y_pred = [estimator.predict_proba_one(x) for estimator in self.ensemble]
        return {c: np.mean([pred.get(c, 0) for pred in y_pred]) for c in set().union(*y_pred)}

# Usage example
from river import datasets
from river import evaluate
from river import metrics
from river import preprocessing

# Create an OESPL instance
oespl = OESPL(
    base_estimator=tree.HoeffdingTreeClassifier(),
    ensemble_size=3,
    lambda_fixed=6.0,
    seed=42,
    drift_detector=drift.ADWIN(),
    patience=1000,
    awakening=500,
    reset_model=True
)

# Create a pipeline
pipeline = preprocessing.StandardScaler() | oespl

# Prepare the dataset
dataset = datasets.Phishing()

# Evaluate the model
metric = metrics.Accuracy()

evaluate.progressive_val_score(
    dataset,
    pipeline,
    metric,
    print_every=1000,
    show_time=True,
    show_memory=True
      )
