from pillar_project.data_utils.dataset import PillarProjectDataframe, Scoreset
from fit import Fit
import json


df = PillarProjectDataframe("/data/dzeiberg/pillar_project/pillar_data_condensed_01_24_25.csv")
ds = Scoreset(df.dataframe[df.dataframe.Dataset == "BRCA1_Adamovich_2022_HDR"])
print(ds)
fit = Fit(ds)
fit.run(core_limit=1, num_fits=1)
result = fit.to_dict()
print(json.dumps(result, indent=4))