from pillar_project.data_utils.dataset import PillarProjectDataframe, Scoreset
from pillar_project.fit_utils.fit import Fit
import json
from pathlib import Path
import datetime
from tqdm import tqdm,trange
import logging

def generate_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


CORELIMIT = 100
NUMFITS = 100
FITS_PER_DATASET = 25
SAVE_DIR = Path(f"/data/dzeiberg/pillar_project/fit_results_{generate_timestamp()}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
logging.info(f"Saving results to {SAVE_DIR}")

df = PillarProjectDataframe("/data/dzeiberg/pillar_project/pillar_data_condensed_01_24_25.csv")
for dataset_name, dataset_df in tqdm(df.dataframe.groupby("Dataset"),total=len(df.dataframe.Dataset.unique())):
    ds = Scoreset(dataset_df,missense_only=True)
    fit = Fit(ds)
    for fit_repetition in trange(FITS_PER_DATASET, desc=dataset_name):
        fit.run(core_limit=CORELIMIT, num_fits=NUMFITS)
        result = fit.to_dict()
        with open(SAVE_DIR / f"{dataset_name}_{generate_timestamp()}.json", "w") as f:
            f.write(json.dumps(result, indent=4))
