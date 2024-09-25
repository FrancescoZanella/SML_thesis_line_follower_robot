import numpy as np
from river import base, tree, drift
import copy

class OESPL(base.Regressor):
    def __init__(self, base_estimator=tree.HoeffdingTreeRegressor(), 
                 ensemble_size=1, lambda_fixed=6.0, seed=1, 
                 drift_detector=drift.ADWIN(), patience=3000, 
                 awakening=1000, reset_model=True):
        # Base learner to train (only tree)
        self.base_estimator = base_estimator  
        # The number of tree base learners in the ensemble
        self.ensemble_size = ensemble_size  
        # Max lambda value to use in Poisson distribution
        self.lambda_fixed = lambda_fixed  
        # Seed value
        self.seed = seed  
        # The change detector strategy to use for drifts
        self.drift_detector = drift_detector  
        # Every how many examples use Page Hinkley statistical test to check whether the base learner trees are statistically growing
        self.patience = patience  
        # For how many samples to apply the Spaced Learning heuristic
        self.awakening = awakening  
        # Should reset a tree base learner after a drift detection?
        self.reset_model = reset_model  
        
        self.ensemble = None
        self.random_state = np.random.RandomState(self.seed)
        self.n_instances = []
        self.drift_detectors = []
        self.growth = []
        self.past_growth = []
        self.awakening_counters = []
        self.patiences = []
        self.page_hinkley = []
       

    def _init_ensemble(self):
        
        #print(f'initializing ensemble')
        self.ensemble = [copy.deepcopy(self.base_estimator) for _ in range(self.ensemble_size)]
        
        for i in range(self.ensemble_size):
            self.page_hinkley.append(drift.PageHinkley(min_instances=3,threshold=0.5))
            self.growth.append(True)
            self.past_growth.append(False)
            self.awakening_counters.append(0)
            self.patiences.append(self.patience)
            self.n_instances.append(0)
            self.drift_detectors.append(copy.deepcopy(self.drift_detector))
        

    def learn_one(self, x, y):
        
        for i in range(self.ensemble_size):
            
            self.n_instances[i] += 1
            
            tree_size = self.ensemble[i].summary['n_nodes'] if self.ensemble[i].summary['n_nodes'] is not None else 0
            
            self.page_hinkley[i].update(tree_size)
            
            
            if self.n_instances[i] % self.patiences[i] == 0:
                
                
                self.past_growth[i] = self.growth[i]

                recent_drifts = self.page_hinkley[i].drift_history[-20:]
                
                self.growth[i] = any(recent_drifts) or self.page_hinkley[i].drift_detected
                if i==0:
                    print(f'growth {self.growth[i]}')              
                if self.growth[i] == False and self.past_growth[i] == False:
                    if i==0:
                        print(f'awakening')
                    self.awakening_counters[i] = self.awakening
                    self.patiences[i] *= 2
                else:
                    if self.growth[i] == True and self.past_growth[i] == False:
                        if i==0:
                            print(f'finished awakening')
                        self.awakening_counters[i] = 0
                        self.patiences[i] = self.patience
            
            if self.growth[i] == True or self.awakening_counters[i] > 0:
                if i==0:
                    print(f'learning')
                lambda_val = self.lambda_fixed
                if self.awakening_counters[i] > 0:
                    self.awakening_counters[i] -= 1
            else:
                if i==0:
                    print(f'not learning')
                lambda_val = 0.1
            
            k = self.random_state.poisson(lambda_val)
            
            if k > 0:
                self.ensemble[i].learn_one(x, y,w=k)
            
            
            # Poisson sampling
            k = self.random_state.poisson(lambda_val)
            
            if k > 0:
                self.ensemble[i].learn_one(x, y,w=k)

            self._drift_detection(x, y, i)
        
        
        
          
        return self

    def _drift_detection(self, x, y, i):

        
        
        self.drift_detectors[i].update(x['sensor0'])
        
        if self.drift_detectors[i].drift_detected:
            print(f'drift detected, resetting ensemble {i}')

            self.page_hinkley[i] = drift.PageHinkley(min_instances=3,threshold=0.5)
            self.n_instances[i] = 0
            self.patiences[i] = self.patience
            self.awakening_counters[i] = 0
            self.growth[i] = True
            self.past_growth[i] = False

            self.drift_detectors[i] = copy.deepcopy(self.drift_detector)
            
            if self.reset_model:
                self.ensemble[i] = copy.deepcopy(self.base_estimator)

    def predict_one(self, x):
        
        if self.ensemble is None:
            self._init_ensemble()
        predictions = [estimator.predict_one(x) for estimator in self.ensemble]
        
        pred = np.mean(predictions)
        return np.mean(pred)

