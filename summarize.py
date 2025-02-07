from pillar_project.data_utils.dataset import PillarProjectDataframe, Scoreset
import json
from fire import Fire
from pathlib import Path
import numpy as np

df = PillarProjectDataframe("/data/dzeiberg/pillar_project/dataframe/pillar_data_condensed_01_28_25.csv")

def summarize(dataset, results_dir,save_dir=None):
    results_dir = Path(results_dir)
    result_files = list(results_dir.glob(f"{dataset}_*.json"))
    results = []
    for result_file in result_files:
        with open(result_file) as f:
            results.append(json.load(f))
    threshold_summary = summarize_thresholds(results)
    fit_quality_summary = summarize_fit_quality(results)
    prior_summary = summarize_priors(results)
    results_summary = {"results_dir" : str(results_dir),
                       "dataset" : dataset,
                       "n_results" : len(results),
                       "prior_summary" : prior_summary,
                       "threshold_summary" : threshold_summary,
                       "fit_quality_summary" : fit_quality_summary}
    print(json.dumps(results_summary,indent=4))
    
    # save_dir = Path(save_dir)
    # save_dir.mkdir(parents=True,exist_ok=True)
    # plot_fit(dataset,results,save_dir)
    # plot_fit_quality(dataset,results,save_dir)

def summarize_priors(results):
    priors = np.array([result['prior'] for result in results])
    return dict(zip(["2.5%", "50%", "97.5%"],np.quantile(priors,[.025,.5,.975])))

def summarize_thresholds(results,minimum_frac=0.5):
    pathogenic_score_thresholds = np.stack([result['score_thresholds']['pathogenic'] for result in results])
    benign_score_thresholds = np.stack([result['score_thresholds']['benign'] for result in results])
    meets_strength_pathogenic = np.isnan(pathogenic_score_thresholds).sum(axis=0) / len(pathogenic_score_thresholds) < minimum_frac
    meets_strength_benign = np.isnan(benign_score_thresholds).sum(axis=0) / len(benign_score_thresholds) < minimum_frac
    scores_pathogenic = np.nanquantile(pathogenic_score_thresholds,0.5,axis=0)
    scores_benign = np.nanquantile(benign_score_thresholds,0.5,axis=0)
    scores_pathogenic[~meets_strength_pathogenic] = np.nan
    scores_benign[~meets_strength_benign] = np.nan
    return {"Pathogenic": dict(zip([1,2,4,8],scores_pathogenic)),"Benign": dict(zip([1,2,4,8],scores_benign))}

def summarize_fit_quality(results):
    cdf_dists = {}
    for result in results:
        for sampleName, sampleFit in result['eval_metrics'].items():
            if sampleName not in cdf_dists:
                cdf_dists[sampleName] = []
            cdf_dists[sampleName].append(sampleFit['cdf_dist'])
    dist_summary = {sampleName: dict(zip(["2.5%","50%", "97.5%"],
                                         (np.quantile(cdf_dists,[.025,.5,.975])))) \
                                            for sampleName,cdf_dists in cdf_dists.items()}
    return dist_summary

def plot_fit(dataset,results,save_dir):
    pass

def plot_fit_quality(dataset,results,save_dir):
    pass

if __name__ == "__main__":
    Fire(summarize)
# summarize("BRCA1_Adamovich_2022_Cisplatin", "/data/dzeiberg/pillar_project/fit_results_20250128_140148/")