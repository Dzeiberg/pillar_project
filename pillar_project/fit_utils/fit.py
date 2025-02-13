from pillar_project.fit_utils.multicomp_model import MulticomponentCalibrationModel
from scipy.stats import skewnorm
import numpy as np
from typing import List, Tuple
from pillar_project.fit_utils.evidence_thresholds import get_tavtigian_constant
import logging
from joblib import Parallel, delayed
import pandas as pd


def tryToFit(observations, sample_indicators, num_components, **kwargs):
    model = MulticomponentCalibrationModel(num_components)
    try:
        model.fit(observations, sample_indicators, **kwargs)
    except Exception as e:
    #     logging.error(f"Failed to fit model\n {e}")
        if not hasattr(model,'_log_likelihoods'):
            model._log_likelihoods = []
        model._log_likelihoods.append(-np.inf)
    return model

def get_bootstrap_indices(dataset_size):
    """
    Generate a bootstrap split of a dataset of a given size

    Required Arguments:
    --------------------------------
    dataset_size -- int
        The size of the dataset to generate the bootstrap split for

    Returns:
    --------------------------------
    train_indices -- np.ndarray
        The indices of the training set
    test_indices -- np.ndarray
        The indices of the test set
    """
    indices = np.arange(dataset_size)
    train_indices = np.random.choice(indices, size=dataset_size, replace=True)
    test_indices = np.setdiff1d(indices, train_indices)
    return train_indices, test_indices

def sample_specific_bootstrap(sample_assignments):
    """
    Bootstrap each sample separately

    Required Arguments:
    --------------------------------
    sample_assignments -- np.ndarray (NSamples, NComponents)
        The one-hot encoded sample assignments
    
    Returns:
    --------------------------------
    train_indices -- np.ndarray
        The indices of the training set
    eval_indices -- np.ndarray
        The indices of the eval set
    """
    train_indices = []
    eval_indices = []
    for sample_num in range(sample_assignments.shape[1]):
        sample_indices = np.where(sample_assignments[:,sample_num])[0]
        if not len(sample_indices):
            continue
        sample_eval = []
        while not len(sample_eval):
            sample_train = np.random.choice(sample_indices, size=len(sample_indices), replace=True)
            sample_eval = np.setdiff1d(sample_indices, sample_train)
        train_indices.append(sample_train)
        eval_indices.append(sample_eval)
    train_indices = np.concatenate(train_indices)
    eval_indices = np.concatenate(eval_indices)
    return train_indices, eval_indices

class Fit:
    def __init__(self, scoreset):
        self.scoreset = scoreset

    def run(self,component_range,**kwargs):
        """
        Run a single fit on a bootstrapped sample of the data
        
        Optional Arguments:
        --------------------------------
        num_fits -- int (default 100)
            The number of fits to run
        core_limit -- int (default -1)
            The number of cores to use for parallel processing
        """
        NUM_FITS = kwargs.get('num_fits',100)
        observations = self.scoreset.scores
        observations = pd.to_numeric(observations, errors='coerce')
        sample_assignments = self.scoreset.sample_assignments
        sample_assignments = makeOneHot(sample_assignments)
        include = sample_assignments.any(axis=1) & ~np.isnan(observations)
        observations = observations[include]
        sample_assignments = sample_assignments[include]
        train_indices , val_indices = sample_specific_bootstrap(sample_assignments)
        train_observations = observations[train_indices]
        train_sample_assignments = sample_assignments[train_indices]
        val_observations = observations[val_indices]
        val_sample_assignments = sample_assignments[val_indices]
        core_limit = kwargs.get('core_limit',-1)
        if core_limit == 1:
            models = [tryToFit(train_observations,train_sample_assignments,num_components, **kwargs) for i in range(NUM_FITS) for num_components in component_range]
        else:
            models = Parallel(n_jobs=kwargs.get('core_limit',-1))(delayed(tryToFit)(train_observations,
                                                                                            train_sample_assignments, num_components, **kwargs) \
                                                                        for i in range(NUM_FITS) for num_components in component_range)
        # models = sorted(models,key=lambda x: x._log_likelihoods[-1],reverse=True)
        val_lls = [m.get_log_likelihood(val_observations,val_sample_assignments) for m in models]
        best_idx = np.nanargmax(val_lls)
        best_fit = models[best_idx]
        if np.isinf(val_lls[best_idx]):
            raise ValueError("Failed to fit model")
        self.model = best_fit
        self._fit_eval()

    def joint_densities(self, x, sampleNum):
        """
        weighted pdfs of a mixture of skew normal distributions

        Parameters
        ----------
        x : np.array (n,)
            values at which to evaluate the pdf
        sampleNum : int
            index of the sample to use
        
        Returns
        -------
        np.array (k, n)
            joint density for each component of the mixture
        """
        weights = self.fit_result['weights'][sampleNum]
        return np.array([w * skewnorm.pdf(x, a, loc, scale) for (a, loc, scale), w in zip(self.fit_result['component_params'], weights)])
    
    def _fit_eval(self):
        """
        Evaluate the fit quality
        """
        self._eval_metrics = {}
        for sampleNum,(sample_scores, sample_name) in enumerate(self.scoreset.samples):
            u = np.unique(sample_scores)
            u.sort()
            self._eval_metrics[sample_name] = {}
            self._eval_metrics[sample_name]['empirical_cdf'] = self.model.empirical_cdf(u)
            self._eval_metrics[sample_name]['model_cdf'] = self.model.get_sample_cdf(u,sampleNum)
            self._eval_metrics[sample_name]['cdf_dist'] = self.model.yang_dist(self._eval_metrics[sample_name]['empirical_cdf'],
                                                                               self._eval_metrics[sample_name]['model_cdf'])

    def scoreset_is_flipped(self):
        """
        Check if the scoreset is flipped
        """
        logging.warning("Unsure if this is applicable for multi-component models")
        _isflipped = self.model.sample_weights[0,0] < self.model.sample_weights[1,0] and self.fit_result['component_params'][0][1] < self.fit_result['component_params'][1][1]
        return _isflipped
    
    def get_prior_estimate(self):
        logging.warning("Unsure if this is applicable for multi-component models")
        return prior_from_weights(self.model.sample_weights, inverted=self.scoreset_is_flipped())
    
    def get_log_lrPlus(self,x, pathogenic_idx=0, controls_idx=1):
        fP = self.model.get_sample_density(x, pathogenic_idx)
        fB = self.model.get_sample_density(x, controls_idx)
        return np.log(fP) - np.log(fB)
    
    def get_score_thresholds(self,point_values):
        logging.warning("Unsure if this is applicable for multi-component models")
        score_thresholds_pathogenic, score_thresholds_benign = calculate_score_thresholds(self.get_log_lrPlus(self.scoreset.scores),
                                                                        self.get_prior_estimate(),
                                                                        self.scoreset.scores,
                                                                        point_values,
                                                                        inverted=self.scoreset_is_flipped())
        return score_thresholds_pathogenic, score_thresholds_benign
    
    def to_dict(self,skip_thresholds=True):
        model_params = {k : v.tolist() for k,v in self.model.get_params().items()}
        extra = {}
        if not skip_thresholds:
            lrPlus_pathogenic, lrPlus_benign = self.get_score_thresholds([1,2,4,8])
            extra = {'prior' : self.get_prior_estimate(),
                 'score_thresholds' : {'pathogenic' : lrPlus_pathogenic.tolist(),
                                       'benign' : lrPlus_benign.tolist()}}
        return {**model_params,**extra,
                'eval_metrics': {k : {'empirical_cdf' : v['empirical_cdf'].tolist(), 'model_cdf' : v['model_cdf'].tolist(), 'cdf_dist': v['cdf_dist']} for k,v in self._eval_metrics.items()},}


def prior_from_weights(weights : np.ndarray, population_idx : int=2, controls_idx : int=1, pathogenic_idx : int=0, inverted: bool = False) -> float:
    """
    Calculate the prior probability of an observation from the population being pathogenic

    Required Arguments:
    --------------------------------
    weights -- Ndarray (NSamples, NComponents)
        The mixture weights of each sample

    Optional Arguments:
    --------------------------------
    population_idx -- int (default 2)
        The index of the population component in the weights matrix
    
    controls_idx -- int (default 1)
        The index of the controls (i.e. benign) component in the weights matrix

    pathogenic_idx -- int (default 0)
        The index of the pathogenic component in the weights matrix

    Returns:
    --------------------------------
    prior -- float
        The prior probability of an observation from the population being pathogenic
    """
    if inverted:
        w_idx = 1
    else:
        w_idx = 0
    prior = ((weights[population_idx, w_idx] - weights[controls_idx, w_idx]) / (weights[pathogenic_idx, w_idx] - weights[controls_idx, w_idx])).item()
    if prior <= 0 or prior >= 1:
        return np.nan
    return prior

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

def calculate_score_thresholds(log_LR,prior,rng,point_values,inverted=False):
    clipped_prior = np.clip(prior,.005,.55) # these seem to be the limits of the tavtigian constant
    lr_thresholds_pathogenic , lr_thresholds_benign = thresholds_from_prior(clipped_prior,point_values)
    log_lr_thresholds_pathogenic = np.log(lr_thresholds_pathogenic)
    log_lr_thresholds_benign = np.log(lr_thresholds_benign)
    pathogenic_score_thresholds = np.ones(len(log_lr_thresholds_pathogenic)) * np.nan
    benign_score_thresholds = np.ones(len(log_lr_thresholds_benign)) * np.nan
    for strength_idx,log_lr_threshold in enumerate(log_lr_thresholds_pathogenic):
        if log_lr_threshold is np.nan:
            continue
        exceed = np.where(log_LR > log_lr_threshold)[0]
        if len(exceed):
            if inverted:
                pathogenic_score_thresholds[strength_idx] = rng[min(exceed)]
            else:
                pathogenic_score_thresholds[strength_idx] = rng[max(exceed)]
    for strength_idx,log_lr_threshold in enumerate(log_lr_thresholds_benign):
        if log_lr_threshold is np.nan:
            continue
        exceed = np.where(log_LR < log_lr_threshold)[0]
        if len(exceed):
            if inverted:
                benign_score_thresholds[strength_idx] = rng[max(exceed)]
            else:
                benign_score_thresholds[strength_idx] = rng[min(exceed)]
    return pathogenic_score_thresholds,benign_score_thresholds

def makeOneHot(sample_assignments):
    assert np.all(sample_assignments.any(axis=0))
    sample_assignments = np.array(sample_assignments)
    onehot = np.zeros_like(sample_assignments)
    cset = np.any(sample_assignments, axis=0)
    while not np.all(np.any(onehot, axis=0)):
        for i in range(sample_assignments.shape[0]):
            true_indices = np.where(sample_assignments[i])[0]
            if len(true_indices) > 0:
                selected_index = np.random.choice(true_indices)
                onehot[i] = False
                onehot[i, selected_index] = True
    assert np.all(np.any(onehot,axis=0))
    assert np.all(onehot.sum(axis=1) <= 1)
    return onehot