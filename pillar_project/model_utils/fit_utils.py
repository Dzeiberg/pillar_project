from mave_calibration.main import single_fit, prior_from_weights
from mave_calibration.skew_normal.density_utils import mixture_pdf, joint_densities, component_posteriors
from mave_calibration.evidence_thresholds import get_tavtigian_constant

import numpy as np
from typing import List, Tuple
import logging
import json
from pillar_project.data_utils.dataset import FunctionalDataset

class Model:
    def __init__(self, dataset : FunctionalDataset):
        self.dataset = dataset
        self.fit_result = None

    @classmethod
    def from_fit_result(cls, dataset : FunctionalDataset, fit_result : dict):
        model = cls(dataset)
        fit_result['weights'] = np.array(fit_result['weights'])
        model.fit_result = fit_result

        model._post_fit(np.arange(dataset.scores.min(), dataset.scores.max(), .001))
        return model
        
    def prepare_sample_assignment(self):
        """
        Each variant can only be included in one sample. Randomly assign variants to one of their samples
        """
        S_multi = self.dataset.sample_assignments
        include_sample = S_multi.sum(0) > 0
        satisfied = False
        while not satisfied:
            S = np.zeros_like(S_multi)
            for i,row in enumerate(S_multi):
                if row.sum() == 0:
                    continue
                S[i,np.random.choice(np.where(row == 1)[0])] = 1
            satisfied = True
            for inc, col in zip(include_sample,S.T):
                if inc and not col.sum():
                    satisfied = False
        return S

    def fit(self, **kwargs):
        """
        Optional Parameters
        -------------------
        - bootstrap : bool (default = False)
            Whether to bootstrap the fit
        """
        S = self.prepare_sample_assignment()
        fit_mask = (S.sum(1) == 1) & (~np.isnan(self.dataset.scores))
        X = self.dataset.scores[fit_mask]
        S = S[fit_mask]
        if kwargs.get("bootstrap",False):
            self._bootstrap_indices = np.random.randint(0,len(X),size=(len(X)))
            X_fit = X[self._bootstrap_indices]
            S_fit = S[self._bootstrap_indices]
        else:
            X_fit = X
            S_fit = S
            self._bootstrap_indices = np.array([])
        self.S_fit = S_fit[:,S_fit.sum(0) > 0]
        
        self.fit_result = single_fit(X_fit, self.S_fit)
        self._post_fit(np.arange(X.min(), X.max(), .001), **kwargs)

    def sample_density(self, X : list, sample_num : int):
        return mixture_pdf(X, self.fit_result.get("component_params"),
                           self.fit_result.get("weights")[sample_num])
    
    @property
    def prior(self):
        if self.fit_result is None:
            raise ValueError("Model has not been fit yet")
        return prior_from_weights(self.fit_result.get("weights"))
    
    def predict_log_lrPlus(self, X : list) -> np.ndarray:
        """
        predict log positive likelihood ratio
        """
        if self.fit_result is None:
            raise ValueError("Model has not been fit yet")
        fP = self.sample_density(X, 0)
        fB = self.sample_density(X, 1)
        return np.log(fP) - np.log(fB)
    
    def predict(self, X : list) -> np.ndarray:
        """
        Assign evidence strengths to each score in X
        """
        if self.fit_result is None:
            raise ValueError("Model has not been fit yet")
        points = np.zeros(len(X))
        for i, x in enumerate(X):
            points[i] = self._assign_point(x)
        return points

    def _assign_point(self, score : float):
        if np.isnan(score):
            return 0
        for threshold,points in list(zip(self.pathogenic_score_thresholds,self._point_values))[::-1]:
            if np.isnan(threshold):
                continue
            if self.inverted and score >= threshold:
                return points
            if (not self.inverted) and score <= threshold:
                return points
        for threshold,points in list(zip(self.benign_score_thresholds,-1 * self._point_values))[::-1]:
            if np.isnan(threshold):
                continue
            if self.inverted and score <= threshold:
                return points
            if (not self.inverted) and score >= threshold:
                return points
        return 0

    def _post_fit(self, score_range : list, **kwargs):
        if self.fit_result is None:
            raise ValueError("Model has not been fit yet")
        # check if lower scores indicate more likely to be pathogenic
        self.inverted = self.fit_result["weights"][0,0] < self.fit_result["weights"][1,0]
        self.pathogenic_score_thresholds, self.benign_score_thresholds = self._calculate_score_thresholds(score_range,**kwargs)

    def _calculate_score_thresholds(self, score_range : list,**kwargs):
        """
        Calculate the score thresholds for each strength of evidence

        Parameters
        ----------
        score_range : list
            The range of scores to consider when assigning thresholds

        Optional Parameters
        -------------------
        point_values : list (default = [1,2,3,4,8])
            The point values for which to assign score thresholds

        Returns
        -------
        pathogenic_score_thresholds : list
            The score thresholds for each strength of pathogenic evidence

        benign_score_thresholds : list
            The score thresholds for each strength of benign evidence
        """
        if self.fit_result is None:
            raise ValueError("Model has not been fit yet")
        self._point_values = np.array(kwargs.get("point_values",[1,2,3,4,8]))
        
        clipped_prior = np.clip(self.prior,.005,.55) # these seem to be the limits of the tavtigian constant
        lr_thresholds_pathogenic , lr_thresholds_benign = Model.thresholds_from_prior(clipped_prior,self._point_values)
        log_lr_thresholds_pathogenic = np.log(lr_thresholds_pathogenic)
        log_lr_thresholds_benign = np.log(lr_thresholds_benign)
        pathogenic_score_thresholds = np.ones(len(log_lr_thresholds_pathogenic)) * np.nan
        benign_score_thresholds = np.ones(len(log_lr_thresholds_benign)) * np.nan
        log_LR = self.predict_log_lrPlus(score_range)
        for strength_idx,log_lr_threshold in enumerate(log_lr_thresholds_pathogenic):
            if log_lr_threshold is np.nan:
                continue
            exceed = np.where(log_LR > log_lr_threshold)[0]
            if len(exceed):
                if self.inverted:
                    pathogenic_score_thresholds[strength_idx] = score_range[min(exceed)]
                else:
                    pathogenic_score_thresholds[strength_idx] = score_range[max(exceed)]
        for strength_idx,log_lr_threshold in enumerate(log_lr_thresholds_benign):
            if log_lr_threshold is np.nan:
                continue
            exceed = np.where(log_LR < log_lr_threshold)[0]
            if len(exceed):
                if self.inverted:
                    benign_score_thresholds[strength_idx] = score_range[max(exceed)]
                else:
                    benign_score_thresholds[strength_idx] = score_range[min(exceed)]
        return pathogenic_score_thresholds,benign_score_thresholds
    
    @staticmethod
    def thresholds_from_prior(prior, point_values) -> Tuple[List[float]]:
        """
        Get the evidence thresholds (LR+ values) for each point value given a prior

        Parameters
        ----------
        prior : float
            The prior probability of pathogenicity

        
        """
        exp_vals = 1 / np.array(point_values).astype(float)
        C,num_successes = get_tavtigian_constant(prior,return_success_count=True)
        # max number of successes is 17
        max_successes = 17
        pathogenic_evidence_thresholds = np.ones(len(point_values)) * np.nan
        benign_evidence_thresholds = np.ones(len(point_values)) * np.nan
        if num_successes < max_successes:
            logging.warning(f"Only ({num_successes})/{max_successes} rules for combining evidence are satisfied by constant {C}, found using prior of ({prior:.4f})")
            return pathogenic_evidence_thresholds, benign_evidence_thresholds
            
        for strength_idx, exp_val in enumerate(exp_vals):
            pathogenic_evidence_thresholds[strength_idx] = C ** exp_val
            benign_evidence_thresholds[strength_idx] = C ** -exp_val
        return pathogenic_evidence_thresholds[::-1], benign_evidence_thresholds[::-1]
    
    def to_json(self):
        if self.fit_result is None:
            raise ValueError("Model has not been fit yet")
        savevals = {
                "component_params": self.fit_result.get('component_params'),
                "weights": self.fit_result.get('weights').tolist(),
                "likelihoods": self.fit_result.get('likelihoods').tolist(),
                "bootstrap_indices": self._bootstrap_indices.tolist(),
                "history": [(hist['component_params'], hist['weights'].tolist()) for hist in self.fit_result.get("history",[])],
                "kmeans_centers": self.fit_result['kmeans'].cluster_centers_.tolist() if 'kmeans' in self.fit_result else None
            }
        for k in savevals:
            if not Model.is_serializable(savevals[k]):
                raise ValueError(f"Value {k} is not serializable")
        return json.dumps(savevals)

    @staticmethod
    def is_serializable(value):
        try:
            json.dumps(value)
            return True
        except TypeError:
            return False