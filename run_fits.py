from pillar_project.data_utils.dataset import PillarProjectDataframe, Scoreset
from pillar_project.fit_utils.fit import Fit
import json
from pathlib import Path
import datetime
from tqdm import tqdm,trange
import logging
import fire
import sys
import joblib
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
def generate_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def run_single_fit(scoreset_id, scoreset_filepath, save_dir,**kwargs):
    """
    Run a single fit on a dataset and save the results to a file
    
    Args:
        dataset_name (str): Name of the dataset to fit
        save_dir (str): Directory to save the fit results
    **kwargs:
        core_limit (int): Number of cores to use (default 32)
        component_range (list): List of number of components to fit (default [2,3])
        num_fits (int): Number of fits to run (default 100)
    """
    # CORELIMIT = kwargs.get("core_limit", 32)
    COMPONENT_RANGE = kwargs.get("component_range", [2,3])
    # NUMFITS = kwargs.get("num_fits", 100)
    save_dir = Path(save_dir)
    
    
    scoreset_filepath = Path(scoreset_filepath)
    if not scoreset_filepath.exists():
        raise FileNotFoundError(f"File {scoreset_filepath} not found")
    scoreset = joblib.load(scoreset_filepath)
    fit = Fit(scoreset)
    fit.run(COMPONENT_RANGE,**kwargs)
    save_dir.mkdir(parents=True, exist_ok=True)
    result = fit.to_dict(skip_thresholds=True)
    with open(save_dir / f"{scoreset_id}_{generate_timestamp()}.json", "w") as f:
        print(f"Saving {scoreset_id} to {f.name}")
        f.write(json.dumps(result, indent=4))

if __name__ == "__main__":
    fire.Fire()