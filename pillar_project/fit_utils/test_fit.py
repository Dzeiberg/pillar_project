from pillar_project.data_utils.dataset import PillarProjectDataframe, Scoreset
from fit import Fit
import json


df = PillarProjectDataframe("/data/dzeiberg/pillar_project/pillar_data_condensed_01_24_25.csv")
for dataset_name, dataset_df in df.dataframe.groupby("Dataset"):
    try:
        ss = Scoreset(dataset_df,missense_only=False)
    except Exception as e:
        print(f"Error in {dataset_name}: {e}")
        continue
    print(dataset_name)
    print(ss)
    print("~~~~~~~~~~~~~~~~")
# ds = Scoreset(df.dataframe[df.dataframe.Dataset == "BARD1_unpublished"])
# print(ds)
# fit = Fit(ds)
# fit.run(core_limit=1, num_fits=1)
# result = fit.to_dict()
# print(json.dumps(result, indent=4))