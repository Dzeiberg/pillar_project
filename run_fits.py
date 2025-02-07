from pillar_project.data_utils.dataset import PillarProjectDataframe, Scoreset
from pillar_project.fit_utils.fit import Fit
import json
from pathlib import Path
import datetime
from tqdm import tqdm,trange
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def generate_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

COMPONENT_RANGE = [2,3]

CORELIMIT = 20
NUMFITS = 5
FITS_PER_DATASET = 12
N_LOOPS = int(5000 / 12) + 1
# N_LOOPS = 1
SAVE_DIR = Path(f"/data/dzeiberg/pillar_project/fit_results_{generate_timestamp()}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Saving results to {SAVE_DIR}")
print(f"Saving results to {SAVE_DIR}")
df = PillarProjectDataframe("/data/dzeiberg/pillar_project/dataframe/pillar_data_condensed_01_28_25.csv")
for loop in trange(N_LOOPS, desc="Loop"):
    for dataset_name, dataset_df in tqdm(df.dataframe.groupby("Dataset"),total=len(df.dataframe.Dataset.unique())):
        ds = Scoreset(dataset_df,missense_only=False)
        if (ds._sample_assignments.sum(0)[:3] == 0).any():
            logger.warning(f"Skipping {dataset_name} because of missing data")
            continue
        fit = Fit(ds)
        for fit_repetition in trange(FITS_PER_DATASET, desc=dataset_name):
            try:
                fit.run(COMPONENT_RANGE, core_limit=CORELIMIT, num_fits=NUMFITS)
            except Exception as e:
                print(f"~~~~~~~~~~~~~~~ FAILED TO FIT {dataset_name} {fit_repetition}")
            #     logger.error(f"Error in {dataset_name} fit {fit_repetition}: {e}")
                continue
            result = fit.to_dict(skip_thresholds=True)
            with open(SAVE_DIR / f"{dataset_name}_{generate_timestamp()}.json", "w") as f:
                logger.info(f"Saving {dataset_name} fit {fit_repetition} to {f.name}")
                f.write(json.dumps(result, indent=4))
                print(f"Saving {dataset_name} fit {fit_repetition} to {f.name}")
