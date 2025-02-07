from pillar_project.fit_utils.multicomp_model import MulticomponentCalibrationModel
from pillar_project.data_utils.dataset import Scoreset, PillarProjectDataframe
import json
from fire import Fire
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def read_scoreset(dataset_path: str, dataset_name: str) -> Scoreset:
    df = PillarProjectDataframe(dataset_path)
    scoreset = Scoreset(df.dataframe[df.dataframe.Dataset == dataset_name])
    return scoreset

def visualize_model(model: MulticomponentCalibrationModel, scoreset: Scoreset, output_path: str):
    distribution_fig = visualize_single_model_distribution(model, scoreset)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    distribution_fig.savefig(output_path / "distribution.png", dpi=300, bbox_inches='tight')
    thresholds_fig = visualize_thresholds(model, scoreset)
    thresholds_fig.savefig(output_path / "thresholds.png", dpi=300, bbox_inches='tight')


def visualize_single_model_distribution(model: MulticomponentCalibrationModel, scoreset: Scoreset):
    n_samples = len(model.sample_weights)
    fig,ax = plt.subplots(n_samples, 1, figsize=(6, 3 * n_samples),sharex=True)
    scores, sample_assignments = scoreset.scores, scoreset.sample_assignments
    sample_names = [t[1] for t in scoreset.samples]
    for sampleNum in range(n_samples):
        sample_scores = scores[sample_assignments[:, sampleNum] == 1]
        sns.histplot(sample_scores, ax=ax[sampleNum],stat='density')
        x = np.linspace(sample_scores.min()-0.25, sample_scores.max() + 0.25, 1000)
        y = model.get_sample_density(x, sampleNum)
        ax[sampleNum].plot(x, y, color='black')
        ax[sampleNum].set_title(sample_names[sampleNum])
    return fig
    
def visualize_thresholds(model: MulticomponentCalibrationModel,scoreset: Scoreset):
    scores = np.arange(scoreset.scores.min(), scoreset.scores.max(), 0.01)
    # lrPlus = model.get_log_lrPlus(scores)
    fP = model.get_sample_density(scores, 0)
    fB = model.get_sample_density(scores, 1)
    lrPlus = np.log(fP) - np.log(fB)
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    ax.plot(scores, lrPlus)
    if model.score_thresholds is not None:
        [ax.axvline(model.score_thresholds['pathogenic'][i], color='red') for i in range(4)]
        [ax.axvline(model.score_thresholds['benign'][i], color='blue') for i in range(4)]
    ax.set_title("lr plus")
    return fig


def reload_model(filepath: str) -> MulticomponentCalibrationModel:
    with open(filepath, 'r') as f:
        data = json.load(f)
    model = MulticomponentCalibrationModel.from_params(data['skewness'], data['locs'], data['scales'], data['sample_weights'])
    try:
        model.score_thresholds = data['score_thresholds']
    except KeyError:
        model.score_thresholds = None
    try:
        model.prior = data['prior']
    except KeyError:
        model.prior = None
    model.eval_metrics = data['eval_metrics']
    return model

def run(model_filepath, dataset_path, dataset_name,output_path):
    model = reload_model(model_filepath)
    scoreset = read_scoreset(dataset_path, dataset_name)
    visualize_model(model, scoreset,output_path)


if __name__ == '__main__':
    Fire(run)
    # run("/data/dzeiberg/pillar_project/fit_results_20250206_155658/CTCF_unpublished_20250206_155847.json",
    #     "/data/dzeiberg/pillar_project/dataframe/pillar_data_condensed_01_28_25.csv",
    #     "CTCF_unpublished",
    #     "/data/dzeiberg/pillar_project/fit_results_20250206_155658/summaries/CTCF_unpublished/")