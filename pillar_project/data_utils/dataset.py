import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np
from pathlib import Path
from fire import Fire
from functools import reduce
import logging
from io import StringIO
import gzip
from tqdm import tqdm
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
        """
        Initialize the scoreset from the dataframe

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to initialize the scoreset from
        
        Returns
        -------
        None
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if len(dataframe.Dataset.unique()) != 1:
            raise ValueError("dataframe must contain only one dataset")
        if not len(dataframe):
            raise ValueError("dataframe must contain at least one row")
        # drop rows with NaN in auth_reported_score
        dataframe = dataframe.assign(auth_reported_score = pd.to_numeric(dataframe.auth_reported_score,errors='coerce'))
        dataframe = dataframe.dropna(subset=["auth_reported_score"])
        dataframe = Scoreset.remove_outliers(dataframe,**kwargs)
        if not len(dataframe):
            raise ValueError("dataframe must contain at least one row with a non-NaN auth_reported_score")
        self.dataframe = dataframe
        self.filter_by_consequence(**kwargs)
        self.variants = [Variant(row) for _, row in self.dataframe.iterrows()]
        self._init_matrices(**kwargs)

    def filter_by_consequence(self,**kwargs):
        self.missense_only = kwargs.get("missense_only",False)
        self.detects_splice = self.dataframe.loc[:,'splice_measure'].unique()[0] == "Yes"
        self.dataframe = self.dataframe[self.dataframe.Flag != "*"]
        if not self.detects_splice:
            self.dataframe = self.dataframe[self.dataframe.simplified_consequence != "Splice Region"]
        if self.missense_only:
            self.dataframe = self.dataframe[self.dataframe.simplified_consequence.isin({'Missense','Synonymous'})]

    @staticmethod
    def remove_outliers(dataframe,**kwargs):
        """
        Optionally clip the dataframe to remove observations outside a specified percentile range

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to remove outliers from

        Optional Parameters
        -------------------
        - quantile_min : float (default: 0.0)
        - quantile_max : float (default: 1.0)
        
        Returns
        -------
        pd.DataFrame
            The dataframe with outliers removed (1.5 IQR Rule)
        """
        quantile_min = kwargs.get("quantile_min",0.0)
        quantile_max = kwargs.get("quantile_max",1.0)
        lowerbound = dataframe.auth_reported_score.quantile(quantile_min)
        upperbound = dataframe.auth_reported_score.quantile(quantile_max)
        scores = dataframe.auth_reported_score
        include = (scores >= lowerbound) & (scores <= upperbound)
        return dataframe[include]

    def __len__(self):
        return len(self.variants)

    def _init_matrices(self,**kwargs):
        self.has_synomyous = any([variant.is_synonymous for variant in self.variants])
        if self.has_synomyous:
            self.NSamples = 4
            self.sample_names = ['Pathogenic/Likely Pathogenic',
                                 'Benign/Likely Benign',
                                 'gnomAD',
                                 'Synonymous']
        else:
            self.NSamples = 3
            self.sample_names = ['Pathogenic/Likely Pathogenic',
                                 'Benign/Likely Benign',
                                 'gnomAD']
        variants_by_id = self.get_variants_by_id()
        self.n_variants = len(variants_by_id)
        self._sample_assignments = np.zeros((self.n_variants,self.NSamples),dtype=bool)
        self._scores = np.zeros(self.n_variants)
        self._ids = []
        self._auth_labels = []
        for idx,(_id,variants) in enumerate(variants_by_id.items()):
            self._ids.append(_id)
            self._scores[idx] = variants[0].auth_reported_score
            self._auth_labels.append(variants[0].auth_reported_func_class)
            if any([variant.is_synonymous for variant in variants]):
                self._sample_assignments[idx,3] = True
                continue
            if any([variant.is_gnomAD for variant in variants]):
                self._sample_assignments[idx,2] = True
            if any([variant.is_pathogenic for variant in variants]):
                self._sample_assignments[idx,0] = True
            if any([variant.is_benign for variant in variants]):
                self._sample_assignments[idx,1] = True
        self.sample_counts = self._sample_assignments.sum(axis=0)

    def get_variants_by_id(self):
        """
        Iterate over all unique Variant.ID values, returning the variants with that given ID.

        Returns
        -------
        dict
            A dictionary where keys are unique Variant.ID values and values are lists of Variant objects with that ID
        """
        variants_by_id = {}
        for variant_id in set(variant.ID for variant in self.variants):
            variants_by_id[variant_id] = [variant for variant in self.variants if variant.ID == variant_id]
        return variants_by_id

    @property
    def sample_assignments(self):
        return self._sample_assignments[:,self.sample_counts > 0]
    
    @property
    def n_samples(self):
        return self.sample_assignments.shape[1]
    
    @property
    def samples(self):
        for sample_index in range(self.NSamples):
            if self.sample_counts[sample_index] > 0:
                yield self.scores[self._sample_assignments[:,sample_index]],self.sample_names[sample_index]
    
    @property
    def scores(self):
        return self._scores
    
    @property
    def scoreset_name(self):
        return self.dataframe.Dataset.values[0]

    def __repr__(self):
        out = f"{self.scoreset_name}: {len(self)} total variants\n"
        for sample_scores, sample_name in self.samples:
            out += f"\t{sample_name}: {len(sample_scores)} variants\n"

        return out
    
class Variant:
    def __init__(self, variant_info: pd.Series):
        self._init_variant_info(variant_info)
    
    def _init_variant_info(self, variant_info: pd.Series):
        for k, v in variant_info.items():
            setattr(self, k, v)
        self.parse_gnomAD_MAF()
        self.parse_clinvar_sig()
        self.parse_consequences()

    def parse_consequences(self):
        self.is_synonymous = self.simplified_consequence == "Synonymous"

    def parse_clinvar_sig(self):
        self.is_conflicting = self.clinvar_sig == "Conflicting classifications of pathogenicity"
        high_quality = self.clinvar_star not in {"no assertion criteria provided",
                                                "no classification for the single variant",
                                                "no classification provided"}
        self.is_benign = high_quality and self.clinvar_sig in {"Benign","Likely benign","Benign/Likely benign"}
        self.is_pathogenic = high_quality and self.clinvar_sig in {"Pathogenic","Likely pathogenic","Pathogenic/Likely pathogenic"} 

    def parse_gnomAD_MAF(self):
        """
        It is possible that the MAF is a list of values separated by a semicolon. If so, parse the list and obtain the maximum value.
        """
        self.is_gnomAD = not pd.isna(self.gnomad_MAF)
    
    @property
    def score(self):
        return self.auth_reported_score
    
    @staticmethod
    def is_nan(value):
        return pd.isna(value) or value == "nan"

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

def csv_to_vcf(input_filepath, output_filepath):
    """
    Convert a CSV file to a gzipped VCF file.

    Parameters
    ----------
    input_filepath : str|Path
        The path to the input CSV file.
    output_filepath : str|Path
        The path to the output gzipped VCF file.

    Returns
    -------
    None
    """
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)

    # Read the CSV file
    df = pd.read_csv(input_filepath)

    # Filter rows with non-null hg38_start
    df = df[df['hg38_start'].notnull()]

    # Open the output file for writing
    with open(output_filepath, 'w') as vcf_file:
        # Write VCF header
        vcf_file.write("##fileformat=VCFv4.2\n")
        vcf_file.write("##source=tsv_to_vcf\n")
        vcf_file.write("""##contig=<ID=1,length=248956422,assembly=GRCh38>
##contig=<ID=2,length=242193529,assembly=GRCh38>
##contig=<ID=3,length=198295559,assembly=GRCh38>
##contig=<ID=4,length=190214555,assembly=GRCh38>
##contig=<ID=5,length=181538259,assembly=GRCh38>
##contig=<ID=6,length=170805979,assembly=GRCh38>
##contig=<ID=7,length=159345973,assembly=GRCh38>
##contig=<ID=8,length=145138636,assembly=GRCh38>
##contig=<ID=9,length=138394717,assembly=GRCh38>
##contig=<ID=10,length=133797422,assembly=GRCh38>
##contig=<ID=11,length=135086622,assembly=GRCh38>
##contig=<ID=12,length=133275309,assembly=GRCh38>
##contig=<ID=13,length=114364328,assembly=GRCh38>
##contig=<ID=14,length=107043718,assembly=GRCh38>
##contig=<ID=15,length=101991189,assembly=GRCh38>
##contig=<ID=16,length=90338345,assembly=GRCh38>
##contig=<ID=17,length=83257441,assembly=GRCh38>
##contig=<ID=18,length=80373285,assembly=GRCh38>
##contig=<ID=19,length=58617616,assembly=GRCh38>
##contig=<ID=20,length=64444167,assembly=GRCh38>
##contig=<ID=21,length=46709983,assembly=GRCh38>
##contig=<ID=22,length=50818468,assembly=GRCh38>
##contig=<ID=X,length=156040895,assembly=GRCh38>
##contig=<ID=Y,length=57227415,assembly=GRCh38>
##contig=<ID=M,length=16569,assembly=GRCh38>
""")
        vcf_file.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        df = df.sort_values(by=['Chrom', 'hg38_start'])
        # Write VCF rows
        for _, row in tqdm(df.iterrows(), total=len(df)):
            vcf_file.write(f"{row['Chrom']}\t{int(row.hg38_start)}\t{row['ID']}\t{row['ref_allele']}\t{row['alt_allele']}\t.\t.\t.\n")

if __name__ == "__main__":
    Fire()
    # summarize_datasets("/data/dzeiberg/pillar_project/dataframe/pillar_data_condensed_gold_standard_02_05_25.csv",missense_only=False, synonymous_exclusive=False,output_file="dataset_summary_all_synonymousNonExclusive.txt")