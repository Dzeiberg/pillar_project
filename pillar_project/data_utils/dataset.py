import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np
from pathlib import Path
from fire import Fire
from functools import reduce
import logging
from io import StringIO
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
class PillarProjectDataframe:
    def __init__(self, data_path: Path|str):
        self.data_path = Path(data_path)
        self.init_data()

    def init_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")
        self.dataframe = pd.read_csv(self.data_path)
    
    def __len__(self):
        return len(self.dataframe)
    
    def get_unique_clinsigs(self):
        sig_sets = self.dataframe.clinvar_sig.apply(lambda li: set(_clean_clinsigs(_tolist(li)))).values
        return reduce(lambda x,y : x.union(y), sig_sets)
        
def _tolist(value,sep="^"):
    try:
        return value.split(sep)
    except AttributeError:
        if pd.isna(value):
            return [np.nan,]
        return [value,]
    
def _clean_clinsigs(values):
    return [v.split(";")[0] if isinstance(v,str) else "nan" for v in values]

class Scoreset:
    def __init__(self, dataframe: pd.DataFrame,**kwargs):
        self._init_dataframe(dataframe,**kwargs)

    def _init_dataframe(self, dataframe : pd.DataFrame,**kwargs):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if len(dataframe.Dataset.unique()) != 1:
            raise ValueError("dataframe must contain only one dataset")
        if not len(dataframe):
            raise ValueError("dataframe must contain at least one row")
        # drop rows with NaN in auth_reported_score
        dataframe = dataframe.dropna(subset=["auth_reported_score"])
        if not len(dataframe):
            raise ValueError("dataframe must contain at least one row with a non-NaN auth_reported_score")
        self.dataframe = dataframe
        variant_type = self.identify_variant_type(dataframe)
        self.variants = [self._init_variant(row,variant_type) for _, row in dataframe.iterrows() \
                         if not pd.isna([row.aa_pos, row.aa_ref, row.aa_alt]).all()]
        self._init_variant_sample_assignments(**kwargs)

    def identify_variant_type(self, dataframe) -> str:
        types = dataframe.nucleotide_or_aa.dropna().unique()
        if len(types) != 1:
            raise ValueError("dataframe must contain only one variant type")
        return types[0]

    def _init_variant(self, row,variant_type):
        return Variant(row)
        # if variant_type == "nucleotide":
        #     return NucleotideVariant(row)
        # elif variant_type == "aa":
        #     return AminoAcidVariant(row)
        # else:
        #     raise ValueError("nucleotide_or_aa must be either 'nucleotide' or 'aa'")

    def __len__(self):
        return len(self.variants)

    def _init_variant_sample_assignments(self,**kwargs):
        NSamples = 4 # P/LP, B/LB, gnomAD, synonymous
        self._sample_assignments = np.zeros((len(self),NSamples), dtype=bool)
        missense_only = kwargs.get("missense_only",False)
        synonymous_exclusive = kwargs.get("synonymous_exclusive",True)
        for i,variant in enumerate(self.variants):
            if variant.has_synonymous():
                self._sample_assignments[i,3] = True
                if synonymous_exclusive:
                    continue
            if variant.has_gnomAD(missense_only):
                self._sample_assignments[i,2] = True
            if variant.has_pathogenic(missense_only):
                self._sample_assignments[i,0] = True
            if variant.has_benign(missense_only):
                self._sample_assignments[i,1] = True

    @property
    def sample_assignments(self):
        return self._sample_assignments[:,self._sample_assignments.sum(axis=0) > 0]
    
    @property
    def n_samples(self):
        return sum(self.sample_assignments.sum(axis=0) > 0)
    
    @property
    def samples(self):
        sample_names = ["Pathogenic/Likely Pathogenic",
                        "Benign/Likely Benign",
                        "gnomAD",
                        "Synonymous"]
        for sample_index in range(4):
            if self._sample_assignments[:,sample_index].sum() > 0:
                yield self.scores[self._sample_assignments[:,sample_index]],sample_names[sample_index]
    
    @property
    def scores(self):
        return np.array([variant.score for variant in self.variants])
    
    def __getitem__(self, idx):
        return self.variants[idx]
    
    def __repr__(self):
        out = f"Scoreset with {len(self)} total variants\n"
        for sample_scores, sample_name in self.samples:
            out += f"\t{sample_name}: {len(sample_scores)} variants\n"

        return out
    
class Variant:
    def __init__(self, variant_info: pd.Series):
        self._init_variant_info(variant_info)
    
    def _init_variant_info(self, variant_info: pd.Series):
        for k, v in variant_info.items():
            setattr(self, k, v)
        self.init_gnomad_MAF()
        # self.validate_variant_info()
        self.parse_clinvar_sig()
        self.parse_consequences()

    def parse_consequences(self):
        synonymous = {'synonymous_variant','synonymous','Synonymous','Silent'}
        missense = {'Missense','missense', 'missense_variant'}
        consequence_values = _tolist(self.consequence)
        flagged_values = [v == "*" for v in _tolist(self.Flag)]
        self.synonymous_annotations = np.array([v in synonymous and not f for v,f in zip(consequence_values,flagged_values)])
        self.missense_annotations = np.array([v in missense and not f for v,f in zip(consequence_values,flagged_values)])

    def parse_clinvar_sig(self):
        self.clinvar_sig_annotations = list(_clean_clinsigs([v for v in _tolist(self.clinvar_sig)]))
        pathogenic = {"Pathogenic","Pathogenic/Likely_pathogenic","Likely_pathogenic"}
        benign = {"Benign","Benign/Likely_benign","Likely_benign"}
        if any([v != "nan" for v in self.clinvar_sig_annotations]):
            logging.info(f"clinvar_sig_annotations: {self.clinvar_sig_annotations}")
        self.pathogenic_annotation = np.array([v in pathogenic for v in self.clinvar_sig_annotations])
        self.benign_annotation = np.array([v in benign for v in self.clinvar_sig_annotations])

    def init_gnomad_MAF(self):
        """
        It is possible that the MAF is a list of values separated by a semicolon. If so, parse the list and obtain the maximum value.
        """
        self.maf_values = pd.to_numeric(_tolist(self.gnomad_MAF),errors='coerce')
    
    def has_pathogenic(self,missense_only):
        if not missense_only:
            return any(self.pathogenic_annotation)
        return any(self.pathogenic_annotation & self.missense_annotations)
        pathogenic_indices = np.where(self.pathogenic_annotation)[0]
        if not missense_only:
            return len(pathogenic_indices) > 0
        missense_indices = np.where(self.missense_annotations)[0]        
        return len(set(missense_indices).intersection(set(pathogenic_indices))) > 0
    
    def has_benign(self,missense_only):
        if not missense_only:
            return any(self.benign_annotation)
        return any(self.benign_annotation & self.missense_annotations)
        benign_indices = np.where(self.benign_annotation)[0]
        if not missense_only:
            return len(benign_indices) > 0
        missense_indices = np.where(self.missense_annotations)[0]
        return len(set(missense_indices).intersection(set(benign_indices))) > 0
    
    def has_synonymous(self):
        return any(self.synonymous_annotations)
    
    def has_gnomAD(self,missense_only):
        if not missense_only:
            return any(self.maf_values > 0)
        return any((self.maf_values > 0) & self.missense_annotations)
        af_indices = np.where(self.maf_values > 0)[0]
        if not missense_only:
            return len(af_indices) > 0
        missense_indices = np.where(self.missense_annotations)[0]
        return len(set(missense_indices).intersection(set(af_indices))) > 0
    
    @property
    def score(self):
        return self.auth_reported_score
    
    @staticmethod
    def is_nan(value):
        return pd.isna(value) or value == "nan"
    
      
# class AminoAcidVariant(Variant):
#     def __eq__(self,other):
#         return self.aa_ref == other.aa_ref and \
#             self.aa_alt == other.aa_alt and \
#             self.aa_pos == other.aa_pos and \
#             self.HGNC_id == other.HGNC_id

# class NucleotideVariant(Variant):
#     def validate_variant_info(self):
#         super().validate_variant_info()
#         return
#         if not hasattr(self, "auth_transcript_id") \
#             or not isinstance(self.auth_transcript_id, str) or \
#                 not len(self.auth_transcript_id):
#             raise ValueError("auth_transcript_id must be a string")
#         if not hasattr(self, "transcript_pos"):
#             raise ValueError("transcript_pos must be provided")
#         if not hasattr(self, "transcript_ref") \
#             or not isinstance(self.transcript_ref, str) or \
#                 self.transcript_ref not in self.valid_nucleotides:
#             raise ValueError(f"transcript_ref must be a valid nucleotide, not {self.transcript_ref} for {self.ID}")
#         if not hasattr(self, "transcript_alt") \
#             or not isinstance(self.transcript_alt, str) or \
#                 self.transcript_alt not in self.valid_nucleotides:
#             raise ValueError(f"transcript_alt must be a valid nucleotide, not {self.transcript_alt} for {self.ID}")
        
#     @property
#     def valid_nucleotides(self):
#         return {'A','C','G','T'}
    
#     def __eq__(self,other):
#         return self.aa_ref == other.aa_ref and \
#             self.aa_alt == other.aa_alt and \
#             self.aa_pos == other.aa_pos and \
#             self.HGNC_id == other.HGNC_id and \
#             self.auth_transcript == other.auth_transcript and \
#             self.transcript_pos == other.transcript_pos and \
#             self.transcript_ref == other.transcript_ref and \
#             self.transcript_alt == other.transcript_alt

def summarize_datasets(dataframe_path, **kwargs):
    """
    Summarize the datasets in the dataframe at dataframe_path.

    Parameters
    ----------
    dataframe_path : str
        The path to the dataframe containing the dataset

    Keyword Arguments
    -----------------
    - output_file : str|Path
        The path to save the summary to

    Returns
    -------
    None
    """
    output_file = kwargs.get("output_file", None)
    if output_file is not None:
        output_file = Path(output_file)
        # output_file.mkdir(parents=True, exist_ok=True)
        f = open(str(output_file),"w")
    else:
        f = StringIO()
    df = PillarProjectDataframe(dataframe_path)
    for dataset_name, ds_df in df.dataframe.groupby("Dataset"):
        scoreset = Scoreset(ds_df,missense_only=kwargs.get("missense_only",False),synonymous_exclusive=kwargs.get("synonymous_exclusive",True))
        f.write(f"{dataset_name}\n")
        f.write(str(scoreset))
        f.write("\n")
    if isinstance(f, StringIO):
        print(f.getvalue())
    else:
        f.close()


if __name__ == "__main__":
    Fire(summarize_datasets)
    # summarize_datasets("/data/dzeiberg/pillar_project/dataframe/pillar_data_condensed_gold_standard_02_05_25.csv",missense_only=False, synonymous_exclusive=False,output_file="dataset_summary_all_synonymousNonExclusive.txt")