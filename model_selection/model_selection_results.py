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
import joblib
from tqdm import tqdm
from typing import Optional,List,Tuple
from joblib import Parallel, delayed
import signal
from random import seed, shuffle
seed(123)
def handler(signum, frame):
    print("Call timed out!")
    raise Exception("Call timed out!")

def reload_results(results_dir, dataset_name):
    """
    Reload results from a directory of fit results
    
    Args:
        results_dir (str): Directory containing fit results
        dataset_name (str): Name of the dataset to reload
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")
        
    results = {}
    files = [file for file in results_dir.rglob("*.json") if dataset_name in file.name]
    for file in tqdm(files, desc="Reloading fit results"):
        with open(file, "r") as f:
            try:
                res = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {file}")
                continue
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
    
    q_vals = [.025, .5, .975]
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
    densities = np.zeros((n_models, n_samples, len(score_range)))
    for i, fit in enumerate(tqdm(fits, desc="Computing densities")):
        for j in range(n_samples):
            densities[i,j] = fit.model.get_sample_density(score_range, j)
    return score_range, densities

def get_thresholds(fits, priors,inverted,point_values=[1,2,3,4,8],**kwargs):
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
        List of point values to use for the thresholds (default [1,2,3,4,8])

    Returns
    ----------
    pathogenic_thresholds : np.ndarray
        Array of shape (n_models, n_point_values) containing the pathogenic score thresholds for each model
    benign_thresholds : np.ndarray
        Array of shape (n_models, n_point_values) containing the benign score thresholds for each model

    """
    n_models = len(fits)
    n_point_values = len(point_values)
    pathogenic_thresholds = np.zeros((n_models, n_point_values))
    benign_thresholds = np.zeros((n_models, n_point_values))
    def compute_thresholds(fit, prior, point_values, inverted):
        return fit.get_score_thresholds(prior, point_values, inverted)
    print("Computing thresholds")
    n_jobs = kwargs.get("n_jobs", -1)
    if n_jobs != 1:
        results = Parallel(n_jobs=-1,verbose=10)(
            delayed(compute_thresholds)(fit, priors[i], point_values, inverted) for i, fit in enumerate(fits)
        )
    else:
        results = []
        for i, fit in enumerate(tqdm(fits, desc="Computing thresholds")):
            results.append(compute_thresholds(fit, priors[i], point_values, inverted))

    for i, (pathogenic, benign) in enumerate(results):
        pathogenic_thresholds[i], benign_thresholds[i] = pathogenic, benign
    return pathogenic_thresholds, benign_thresholds
    
def summarize_thresholds(pathogenic_thresholds : np.ndarray,
                         benign_thresholds : np.ndarray,
                         final_quantile : float,
                         min_exceeding : float,
                         inverted : bool) -> Tuple[np.ndarray, np.ndarray]:
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
    meets_pathogenic = np.isnan(pathogenic_thresholds).sum(axis=0)/len(pathogenic_thresholds) < (1 - min_exceeding)
    meets_benign = np.isnan(benign_thresholds).sum(axis=0)/len(benign_thresholds) < (1 - min_exceeding)
    if inverted:
        QP = 1 - final_quantile
        QB = final_quantile
    else:
        QP = final_quantile
        QB = 1 - final_quantile
    P = np.nanquantile(pathogenic_thresholds, QP, axis=0)
    B = np.nanquantile(benign_thresholds, QB, axis=0)
    P[~meets_pathogenic] = np.nan
    B[~meets_benign] = np.nan
    return P,B
    

def plotDataset(scoreset : Scoreset,
                score_range : np.ndarray,
                densities : np.ndarray,
                priors : np.ndarray,
                pathogenic_thresholds : np.ndarray,
                benign_thresholds : np.ndarray,**kwargs):
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

    Optional Arguments
    ----------
    - subset range : bool
        Whether to subset the range of scores to plot (default False)
    - range_q_min : float
        Minimum quantile to use for the range (default 0.05)
    - range_q_max : float
        Maximum quantile to use for the range (default 0.95)
    
    Returns
    ----------
    fig : matplotlib.Figure
        Figure object containing the plot
    """
    if kwargs.get("subset_range", True):
        range_q_min = kwargs.get("range_q_min", 0.001)
        range_q_max = kwargs.get("range_q_max", 0.999)
        score_limits = np.quantile(scoreset.scores, [range_q_min, range_q_max])
        keep = (score_range >= score_limits[0]) & (score_range <= score_limits[1])
        score_range = score_range[keep]
        densities = densities[:,:,keep]
    else:
        score_limits = [min(scoreset.scores), max(scoreset.scores)]
    n_samples= densities.shape[1]
    fig, ax = plt.subplots(n_samples, 1, figsize=(10, 5*n_samples))
    for sample_num,(sample_scores, sample_name) in enumerate(scoreset.samples):
        sample_densities = densities[:,sample_num]
        lower,median,upper = np.quantile(sample_densities, [.05,.5,.95], axis=0)
        ax[sample_num].fill_between(score_range, lower, upper, alpha=0.5, color='blue')
        ax[sample_num].plot(score_range, median, color='blue')
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)
        sample_scores = sample_scores[(sample_scores >= score_limits[0]) & (sample_scores <= score_limits[1])]
        try:
            sns.histplot(sample_scores,ax=ax[sample_num], stat='density')
        except Exception as e:
            print(e)
            ax[sample_num].hist(sample_scores, density=True)
        signal.alarm(0)
        title = sample_name
        if sample_name == "gnomAD":
            median_prior,lower,upper = np.quantile(priors,[.5, .025, .975])
            title += f" (prior: {median_prior:.3f} [{lower:.3f}, {upper:.3f}])"
        ax[sample_num].set_title(title)
        for sP,sB,ls in zip(pathogenic_thresholds, benign_thresholds,[":","-.","--","-"]):
            ax[sample_num].axvline(sB,linestyle=ls,color='blue')
            ax[sample_num].axvline(sP,linestyle=ls,color='red')
        ax[sample_num].set_xlim(*score_limits)
    return fig
    


def summarize_dataset_fits(fits_dir : str|Path,
                           dataset_name : str,
                           scoresets_dir : str|Path,
                           summary_dir : str|Path,
                           **kwargs):
    try:
        scoreset, fits,component_fit_counts = load(fits_dir, dataset_name, scoresets_dir, **kwargs)
    except ValueError:
        print(f"No fits for {dataset_name} in {fits_dir}")
        return
    score_range, densities = get_sample_densities(scoreset, fits)
    pathogenic_scores = scoreset.scores[scoreset.sample_assignments[:,0] == 1]
    benign_scores = scoreset.scores[scoreset.sample_assignments[:,1] == 1]
    inverted = pathogenic_scores.mean() > benign_scores.mean()
    priors = get_priors(scoreset, fits)

    pathogenic_thresholds = np.ones((len(fits),4))*np.nan
    benign_thresholds = np.ones((len(fits),4))*np.nan
    pathogenic_thresholds, benign_thresholds = get_thresholds(fits, priors,inverted,**kwargs)

    final_quantile = kwargs.get("final_quantile", 0.05)
    min_exceeding = kwargs.get("min_exceeding", 0.95)
    
    
    final_pathogenic_thresholds, final_benign_thresholds = summarize_thresholds(pathogenic_thresholds, benign_thresholds,
                                                                                final_quantile,min_exceeding,inverted)
    
    fig = plotDataset(scoreset, score_range, densities, priors, final_pathogenic_thresholds, final_benign_thresholds,**kwargs)
    summary_dict = make_summary_dict(dataset_name,fits,priors)
    summary_dict['component_fit_counts'] = component_fit_counts
    summary_dict['pathogenic_thresholds'] = list(final_pathogenic_thresholds)
    summary_dict['benign_thresholds'] = list(final_benign_thresholds)
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(summary_dir/f"{dataset_name}_summary.png",dpi=300,bbox_inches='tight')
    with open(summary_dir/f"{dataset_name}_summary.json", "w") as f:
        json.dump(summary_dict, f,indent=4)
    print("Saved summary to", summary_dir/f"{dataset_name}_summary.json")

def make_summary_dict(dataset_name : str, fits : List[Fit], priors : List[float]) -> dict:
    """
    Make a summary dictionary for a dataset

    Arguments
    ----------
    dataset_name : str
        Name of the dataset
    fits : list[Fit]
        List of Fit objects for the dataset
    priors : list[float]
        List of prior estimates for the dataset
    
    Returns
    ----------
    dict
        Dictionary containing the summary information for the dataset
    """
    summary_dict = {}
    summary_dict['scoreset_id'] = dataset_name
    summary_dict['calibration_method'] = "Multi-sample skew normal mixture calibration"
    summary_dict['n_results'] = len(fits)
    prior_summary = summarize_prior_distribution(priors)
    summary_dict['prior_summary'] = {f"{k:g}" : prior_summary[k] for k in [.025,.5,.975]}
    fit_quality = {}
    for fit in fits:
        for sample_name, sample_eval in fit._eval_metrics.items():
            if sample_name not in fit_quality:
                fit_quality[sample_name] = []
            fit_quality[sample_name].append(sample_eval['cdf_dist'])
    quality_summary = {sample_name : {f"{q:g}" : np.quantile(fit_quality[sample_name], q) for q in [.025,.5,.975]} \
                       for sample_name in fit_quality}
    summary_dict['fit_quality_summary'] = quality_summary
    return summary_dict

def load(fits_dir : str|Path,
         dataset_name : str,
         scoresets_dir : str|Path,**kwargs)->tuple[Scoreset, List[Fit]]:
    """
    Load dataset and model fits

    Arguments
    ----------
    fits_dir: str|Path
        Directory containing model fits
    dataset_name: str
        Name of the dataset to summarize
    scoresets_dir: str|Path
       Directory containing the pre-processed scoresets
    n_components: Optional[int]
        Only load results for models with this number of components (default None, meaning load all models) (incompatible with best_n_components)
    best_n_components: Optional[bool]
        Only load results for the best number of components (default False, meaning load all models) (incompatible with n_components); prior value is 2 components, if a multi-component model gets at least 95% of the fits, it will be used
    
    Returns
    ----------
    scoreset : Scoreset
        Scoreset object for the dataset
    fits : list[Fit]
        List of Fit objects for the
    """
    n_components = kwargs.get("n_components", None)
    best_n_components = kwargs.get("best_n_components", None)
    if n_components is not None and best_n_components is not None:
        raise ValueError("n_components and best_n_components are incompatible")
    results = reload_results(fits_dir, dataset_name)
    if len(results) == 0:
        raise ValueError(f"No results found for {dataset_name} in {fits_dir}")
    counts = {k: len(v) for k,v in results.items()}
    print(counts)
    if n_components is not None:
        result_lists = results[n_components]
    elif best_n_components:
        n_components = 2
        for n_comp in sorted(list(set(results.keys()) - {2,}))[::-1]:
            if counts[n_comp] > 0.95*sum(counts.values()):
                n_components = n_comp
        result_lists = results[n_components]
        print(f"Using {n_components} components for {dataset_name}")
    else:
        result_lists = [res for res_list in results.values() for res in res_list]
    scoreset = joblib.load(Path(scoresets_dir)/f"{dataset_name}.pkl")
    n_fits_to_load = kwargs.get("n_fits_to_load", None)
    shuffle(result_lists)
    if n_fits_to_load < 0:
        n_fits_to_load = len(result_lists)
    elif len(result_lists) < n_fits_to_load:
        print(f"Warning: Only loading the {len(result_lists)} fits available for {dataset_name}, requesting {n_fits_to_load}")
    fits = result_lists[:n_fits_to_load] if n_fits_to_load is not None else result_lists
    fits = [Fit.from_dict(scoreset,res) for res in fits]
    return scoreset, fits, counts
    

if __name__ == "__main__":
    fire.Fire()
    # summarize_dataset_fits("files/preprint_fits_K2_03282025/",
    #                        "ASPA_Grønbæk-Thygesen_2024_abundance",
    #                        "files/dataframe/final_scoresets",
    #                        "files/preprint_fits_K2_03282025_summaries/",
    #                        min_exceeding=0.95,final_quantile=0.05,
    #                        n_jobs=1)