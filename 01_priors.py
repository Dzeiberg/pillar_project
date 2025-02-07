#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from typing import List,Tuple


# In[2]:


from pillar_project.fit_utils.multicomp_model import MulticomponentCalibrationModel
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed


# In[3]:


def generate_data(observations_per_sample,**params)->Tuple[np.ndarray, np.ndarray]:
    observations = np.zeros(0)
    sample_indicators = np.zeros((0,params['weights'].shape[0]))
    for sampleNum in range(params['weights'].shape[0]):
        for compNum in range(params['weights'].shape[1]):
            n = np.round(observations_per_sample[sampleNum] * params['weights'][sampleNum, compNum]).astype(int)
            observations = np.concatenate((observations,stats.skewnorm.rvs(params['skewness'][compNum], loc=params['locs'][compNum], scale=params['scales'][compNum], size=n)))
            I = np.zeros((n,params['weights'].shape[0]))
            I[:,sampleNum] = 1
            sample_indicators = np.concatenate([sample_indicators, I])
    return observations, sample_indicators


# In[4]:


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


# In[5]:


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


# In[6]:


import seaborn as sns
def visualize_single_model_distribution(model: MulticomponentCalibrationModel, scores, sample_assignments):
    n_samples = len(model.sample_weights)
    fig,ax = plt.subplots(n_samples, 1, figsize=(6, 3 * n_samples),sharex=True)
    sample_names = ["P/LP", "B/LB",'gnomAD']
    score_range = np.linspace(scores.min()-1, scores.max()+1, 1000)
    for sampleNum in range(n_samples):
        sample_scores = scores[sample_assignments[:, sampleNum] == 1]
        sns.histplot(sample_scores, ax=ax[sampleNum],stat='density')
        y = model.get_sample_density(score_range, sampleNum)
        ax[sampleNum].plot(score_range, y, color='black')
        ax[sampleNum].set_title(sample_names[sampleNum])
    return fig


# In[8]:


def timestamp():
    import time
    return time.strftime("%Y%m%d-%H%M%S")


# In[ ]:

starttime = timestamp()
from pathlib import Path
NRep = 100
NComponents = 2
CORE_LIMIT = 32
NFITS = 32
figdir = Path("test_figs/")
figdir.mkdir(exist_ok=True)
res_file = figdir / f"results_{starttime}.json"
for rep in range(NRep):
    params = dict(
        skewness = np.random.uniform(-2,2,size=NComponents),
        locs = [-5, 5],
        scales = np.random.uniform(1,3,size=NComponents),
        prior = np.random.uniform(0,1),
        weights = np.random.dirichlet(np.ones(NComponents),2)
    )

    params['weights'] = np.concatenate((params['weights'],(params['prior'] * params['weights'][0] + (1-params['prior']) * params['weights'][1]).reshape(1,-1)))

    observations, sample_indicators = generate_data([25,25,100],**params)
    train_indices, val_indices = get_bootstrap_indices(observations.shape[0])
    train_observations = observations[train_indices]
    train_indicators = sample_indicators[train_indices]
    val_observations = observations[val_indices]
    val_indicators = sample_indicators[val_indices]
    models = Parallel(n_jobs=CORE_LIMIT)(delayed(tryToFit)(train_observations,train_indicators, NComponents, max_iters=10000) for i in range(NFITS))
    val_lls = [m.get_log_likelihood(val_observations,val_indicators) for m in models]
    best_idx = np.nanargmax(val_lls)
    if np.isnan(val_lls[best_idx]) or np.isinf(val_lls[best_idx]):
        continue
    model = models[best_idx]
    fig = visualize_single_model_distribution(model, observations, sample_indicators)
    fig.savefig(figdir / f"rep_{timestamp()}.png")
    prior0 = (model.sample_weights[2,0] - model.sample_weights[1,0]) / (model.sample_weights[0,0] - model.sample_weights[1,0])

    xG = observations[sample_indicators[:,2] == 1]
    gP = model.get_sample_density(xG, 0)
    gB = model.get_sample_density(xG, 1)
    alpha = .5
    converged = False
    while not converged:
        # print(alpha)
        posts = 1 / (1 + ((1-alpha) / alpha) * (gB / gP))
        alpha_new = np.mean(posts)
        converged = np.abs(alpha - alpha_new) < 1e-10
        alpha = alpha_new

    # reload results
    if res_file.exists():
        import json
        with res_file.open('r') as f:
            results = json.load(f)
        results['weight_method_err'].append(params['prior'] - prior0)
        results['mlls_err'].append(params['prior'] - alpha)
        results['prior'].append(params['prior'])
    else:
        results = {'weight_method_err': [params['prior'] - prior0], 'mlls_err': [params['prior'] - alpha], 'prior': [params['prior'],]}
    with res_file.open('w') as f:
        import json
        json.dump(results, f)

    print(f"iteration: {rep}\n--------\nMAE weight method: {np.mean(np.abs(results['weight_method_err']))}\nMAE MLLS: {np.mean(np.abs(results['mlls_err']))}\n")



# In[ ]:




