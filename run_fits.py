from pillar_project.data_utils.dataset import PillarProjectDataframe, Scoreset
from pillar_project.fit_utils.fit import Fit
import json
from pathlib import Path
import datetime
from tqdm import tqdm,trange
import logging
import fire
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def generate_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# # parameters
# COMPONENT_RANGE = [2,3]
# CORELIMIT = 20
# NUMFITS = 5
# FITS_PER_DATASET = 12
# N_LOOPS = int(5000 / 12) + 1
# # Init save directory
# SAVE_DIR = Path(f"/data/dzeiberg/pillar_project/fit_results_{generate_timestamp()}")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)
# logger.info(f"Saving results to {SAVE_DIR}")
# print(f"Saving results to {SAVE_DIR}")
# # Read data
# df = PillarProjectDataframe("/data/dzeiberg/pillar_project/dataframe/pillar_data_condensed_01_28_25.csv")

# for loop in trange(N_LOOPS, desc="Loop"):
#     for dataset_name, dataset_df in tqdm(df.dataframe.groupby("Dataset"),total=len(df.dataframe.Dataset.unique())):
#         ds = Scoreset(dataset_df,missense_only=False)
#         if (ds._sample_assignments.sum(0)[:3] == 0).any():
#             logger.warning(f"Skipping {dataset_name} because of missing data")
#             continue
#         fit = Fit(ds)
#         for fit_repetition in trange(FITS_PER_DATASET, desc=dataset_name):
#             try:
#                 fit.run(COMPONENT_RANGE, core_limit=CORELIMIT, num_fits=NUMFITS)
#             except Exception as e:
#                 print(f"~~~~~~~~~~~~~~~ FAILED TO FIT {dataset_name} {fit_repetition}")
#             #     logger.error(f"Error in {dataset_name} fit {fit_repetition}: {e}")
#                 continue
#             result = fit.to_dict(skip_thresholds=True)
#             with open(SAVE_DIR / f"{dataset_name}_{generate_timestamp()}.json", "w") as f:
#                 logger.info(f"Saving {dataset_name} fit {fit_repetition} to {f.name}")
#                 f.write(json.dumps(result, indent=4))
#                 print(f"Saving {dataset_name} fit {fit_repetition} to {f.name}")

def run_single_fit(dataset_name, save_dir,**kwargs):
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
    CORELIMIT = kwargs.get("core_limit", 32)
    COMPONENT_RANGE = kwargs.get("component_range", [2,3])
    NUMFITS = kwargs.get("num_fits", 100)
    save_dir = Path(save_dir)
    

    df = PillarProjectDataframe("/data/dzeiberg/pillar_project/dataframe/pillar_data_condensed_01_28_25.csv")
    dataset_df = df.dataframe[df.dataframe.Dataset == dataset_name]
    ds = Scoreset(dataset_df,missense_only=False)
    fit = Fit(ds)
    fit.run(COMPONENT_RANGE, core_limit=CORELIMIT, num_fits=NUMFITS)
    save_dir.mkdir(parents=True, exist_ok=True)
    result = fit.to_dict(skip_thresholds=True)
    with open(save_dir / f"{dataset_name}_{generate_timestamp()}.json", "w") as f:
        logger.info(f"Saving {dataset_name} to {f.name}")
        f.write(json.dumps(result, indent=4))
        print(f"Saving {dataset_name} to {f.name}")

if __name__ == "__main__":
    fire.Fire()