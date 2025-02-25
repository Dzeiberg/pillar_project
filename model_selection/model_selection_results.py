import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import fire
import sys
sys.path.append("/home/dzeiberg/pillar_project")
from pillar_project.data_utils.dataset import PillarProjectDataframe, Scoreset
from pillar_project.fit_utils.fit import Fit
import os
from tqdm import tqdm
from typing import Optional,List,Tuple

def reload_results(results_dir, dataset_name):
    """
    Reload results from a directory of fit results
    
    Args:
        results_dir (str): Directory containing fit results
        dataset_name (str): Name of the dataset to reload
    """
    results_dir = Path(results_dir)
    results = {}
    files = list(results_dir.glob(f"**/*{dataset_name}*.json"))
    for file in files:
        with open(file, "r") as f:
            res = json.load(f)
        n_components = len(res['skewness'])
        if n_components not in results:
            results[n_components] = []
        results[n_components].append(res)
    return results

def summarize_model_selection(results_dir):
    """
    Summarize model selection results
    
    Args:
        results_dir (str): Directory containing fit results
    """
    results_dir = Path(results_dir)
    dataset_names = set([d[:d.rfind("_")].replace("fit_results_","") for d in os.listdir(results_dir)])
    for dataset in dataset_names:
        results = reload_results(results_dir, dataset)
        print(f"{dataset}:")
        for n_components, res in results.items():
            print(f"  {n_components} : {len(res)}")

def summarize_prior_distribution(priors):
    """
    Summarize the distribution of prior estimates for models fit to a dataset

    Arguments
    ----------
    priors : list[float]
        List of prior estimates
    
    Returns
    ----------
    dict
        Dictionary with keys as quantiles and values as the prior estimates at those quantiles
    """
    
    q_vals = [.05, .5, .95]
    quantiles = np.nanquantile(priors, q_vals)
    ret = dict(zip(q_vals, quantiles))
    ret['num_NaN'] = sum(np.isnan(priors))
    ret['num_fits'] = len(priors)
    return ret

def get_priors(scoreset, fits):
     population_scores = scoreset.scores[scoreset.sample_assignments[:,2] == 1]
     priors = [fit.get_prior_estimate(population_scores) for fit in tqdm(fits, desc="Estimating priors")]
     return priors

def get_sample_densities(scoreset, fits):
    """
    Get the distribution of model densities for each sample in the dataset

    Arguments
    ----------
    scoreset : Scoreset
        Scoreset object for the dataset
    fits : list[Fit]
        List of Fit objects for the dataset
    
    Returns
    ----------
    score_range : np.ndarray
        Array of shape (n_score_range,) containing the range of scores to compute densities over
    densities : np.ndarray
        Array of shape (n_models, n_samples, n_score_range) containing the densities for each model and sample
    """
    score_range = np.arange(scoreset.scores.min(), scoreset.scores.max(), 0.01)
    n_samples = scoreset.n_samples
    n_models = len(fits)
    densities = np.zeros(n_models, n_samples, len(score_range))
    for i, fit in enumerate(tqdm(fits, desc="Computing densities")):
        for j in range(n_samples):
            densities[i,j] = fit.model.get_sample_density(score_range, j)
    return score_range, densities

def get_thresholds(fits, priors, point_values=[1,2,4,8]):
    """
    Get the thresholds for each model

    Arguments
    ----------
    fits : list[Fit]
        List of Fit objects for the dataset
    priors : list[float]
        List of prior estimates for the dataset
    
    Optional Arguments
    ----------
    point_values : list[float]
        List of point values to use for the thresholds (default [1,2,4,8])

    Returns
    ----------
    pathogenic_thresholds : np.ndarray
        Array of shape (n_models, n_point_values) containing the pathogenic score thresholds for each model
    benign_thresholds : np.ndarray
        Array of shape (n_models, n_point_values) containing the benign score thresholds for each model

    """
    n_models = len(fits)
    n_point_values = len(point_values)
    pathogenic_thresholds = np.zeros(n_models, n_point_values)
    benign_thresholds = np.zeros(n_models, n_point_values)
    for i, fit in enumerate(tqdm(fits, desc="Computing thresholds")):
        pathogenic_thresholds[i], benign_thresholds[i] = fit.get_thresholds(priors[i], point_values)
    return pathogenic_thresholds, benign_thresholds
    
def summarize_thresholds(pathogenic_thresholds : np.ndarray,
                         benign_thresholds : np.ndarray,
                         final_quantile : float,
                         min_exceeding : float,
                         flipped : bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the summary of the thresholds computed for each model

    Arguments
    ----------
    pathogenic_thresholds : np.ndarray
        Array of shape (n_models, n_point_values) containing the pathogenic score thresholds for each model
    benign_thresholds : np.ndarray
        Array of shape (n_models, n_point_values) containing the benign score thresholds for each model
    final_quantile : float
        Quantile to use for the final threshold [0,1]
    min_exceeding : float
        Minimum fraction of models reaching the given evidence strength to use the final threshold
    flipped : bool
        Whether the score set is 'flipped' from its canonical orientation

    Returns
    ----------
    final_pathogenic_thresholds : np.ndarray
        Array of shape (n_point_values,) containing the final pathogenic score thresholds
    final_benign_thresholds : np.ndarray
        Array of shape (n_point_values,) containing the final benign score thresholds
    """
    final_pathogenic_thresholds = np.zeros(pathogenic_thresholds.shape[1])
    final_benign_thresholds = np.zeros(benign_thresholds.shape[1])
    meets_pathogenic = np.isnan(pathogenic_thresholds,axis=0) < min_exceeding
    meets_benign = np.isnan(benign_thresholds,axis=0) < min_exceeding
    if flipped:
        QP = 1 - final_quantile
        QB = final_quantile
    else:
        QP = final_quantile
        QB = 1 - final_quantile
    P = np.quantile(pathogenic_thresholds, QP, axis=0)
    B = np.quantile(benign_thresholds, QB, axis=0)
    final_pathogenic_thresholds[meets_pathogenic] = P[meets_pathogenic]
    final_benign_thresholds[meets_benign] = B[meets_benign]
    return final_pathogenic_thresholds, final_benign_thresholds
    

def plotDataset(scoreset : Scoreset,
                score_range : np.ndarray,
                densities : np.ndarray,
                priors : np.ndarray,
                pathogenic_thresholds : np.ndarray,
                benign_thresholds : np.ndarray):
    """
    Plot the dataset summary

    Arguments
    ----------
    scoreset : Scoreset
        Scoreset object for the dataset
    score_range : np.ndarray
        Array of shape (n_score_range,) containing the range of scores to compute densities over
    densities : np.ndarray
        Array of shape (n_models, n_samples, n_score_range) containing the densities for each model and sample
    priors : np.ndarray
        Array of shape (n_models,) containing the prior estimates for each model
    pathogenic_thresholds : np.ndarray
        Array of shape (n_point_values,) containing the pathogenic score thresholds
    benign_thresholds : np.ndarray
        Array of shape (n_point_values,) containing the benign score thresholds
    
    Returns
    ----------
    fig : matplotlib.Figure
        Figure object containing the plot
    """
    n_samples= densities.shape[1]
    fig, ax = plt.subplots(n_samples, 1, figsize=(10, 5*n_samples))
    for sample_num,(sample_scores, sample_name) in enumerate(scoreset.samples):
        sample_densities = densities[:,sample_num]
        lower,median,upper = np.quantile(sample_densities, [.05,.5,.95], axis=0)
        ax[sample_num].fill_between(score_range, lower, upper, alpha=0.5, color='blue')
        ax[sample_num].plot(score_range, median, color='blue')
        sns.histplot(sample_scores,ax=ax[sample_num], stat='density')
        title = sample_name
        if sample_name == "gnomAD":
            median_prior,lower,upper = np.quantile(priors,[.5, .025, .975])
            title += f" (prior: {median_prior:.3f} [{lower:.3f}, {upper:.3f}])"
        ax[sample_num].set_title(title)
        for sP,sB,ls in zip(pathogenic_thresholds, benign_thresholds,[":","-.","--","-"]):
            ax[sample_num].axvline(sB,linestyle=ls,color='blue')
            ax[sample_num].axvline(sP,linestyle=ls,color='red')
    return fig
    


def summarize_dataset_fits(results_dir : str|Path,
                           dataset_name : str,
                           dataset_filepath : str|Path,
                           **kwargs):
    scoreset, fits = load(results_dir, dataset_name, dataset_filepath, **kwargs)
    score_range, densities = get_sample_densities(scoreset, fits)
    priors = get_priors(scoreset, fits)
    pathogenic_thresholds, benign_thresholds = get_thresholds(fits, priors)
    quantile = kwargs.get("final_quantile", 0.05)
    min_exceeding = kwargs.get("min_exceeding", 0.95)
    medianLRPlus = np.quantile(np.log(densities[:,0] - np.log(densities[:,1])), 0.5, axis=0)
    
    final_pathogenic_thresholds, final_benign_thresholds = summarize_thresholds(pathogenic_thresholds, benign_thresholds, **kwargs)
    
    fig = plotDataset(scoreset, score_range, densities, priors, pathogenic_thresholds, benign_thresholds)

def load(results_dir : str|Path,
                                 dataset_name : str,
                                 dataset_filepath : str|Path,**kwargs)->tuple[Scoreset, List[Fit]]:
    """
    Load dataset and model fits

    Arguments
    ----------
    results_dir: str|Path
        Directory containing fit results
    dataset_name: str
        Name of the dataset to summarize
    dataset_filepath: str|Path
        Filepath to the pillar project dataframe
    n_components: Optional[int]
        Only load results for models with this number of components (default None, meaning load all models)
    
    Returns
    ----------
    scoreset : Scoreset
        Scoreset object for the dataset
    fits : list[Fit]
        List of Fit objects for the
    """
    n_components = kwargs.get("n_components", None)
    results = reload_results(results_dir, dataset_name)
    if n_components is not None:
        result_dicts = results[n_components]
    else:
        result_dicts = [res for res_list in results.values() for res in res_list]
    df = PillarProjectDataframe(dataset_filepath)
    scoreset = Scoreset(df.dataframe[df.dataframe.Dataset == dataset_name], missense_only=False)
    fits = [Fit.from_dict(scoreset,res) for res in result_dicts]
    return scoreset, fits
    

if __name__ == "__main__":
    fire.Fire()