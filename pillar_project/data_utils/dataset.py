import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np
from pathlib import Path

from functools import reduce
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
    return [v.split(";")[0] for v in values]

class Scoreset:
    def __init__(self, dataframe: pd.DataFrame,**kwargs):
        self._init_dataframe(dataframe)

    def _init_dataframe(self, dataframe : pd.DataFrame,**kwargs):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if len(dataframe.Dataset.unique()) != 1:
            raise ValueError("dataframe must contain only one dataset")
        if not len(dataframe):
            raise ValueError("dataframe must contain at least one row")
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
        if variant_type == "nucleotide":
            return NucleotideVariant(row)
        elif variant_type == "aa":
            return AminoAcidVariant(row)
        else:
            raise ValueError("nucleotide_or_aa must be either 'nucleotide' or 'aa'")

    def __len__(self):
        return len(self.variants)

    def _init_variant_sample_assignments(self,**kwargs):
        NSamples = 4 # P/LP, B/LB, gnomAD, synonymous
        self._sample_assignments = np.zeros((len(self),NSamples), dtype=bool)
        missense_only = kwargs.get("missense_only",False)
        for i,variant in enumerate(self.variants):
            if variant.is_synonymous:
                self._sample_assignments[i,3] = True
                continue
            if not missense_only or variant.is_missense:
                if variant.present_in_gnomAD:
                    self._sample_assignments[i,2] = True
                if variant.is_pathogenic_or_likely_pathogenic:
                    self._sample_assignments[i,0] = True
                if variant.is_benign_or_likely_benign:
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
        out = f"Scoreset with {len(self)} variants\n"
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
        self.validate_variant_info()
        self.parse_clinvar_sig()

    def parse_clinvar_sig(self):
        if hasattr(self, "clinvar_sig"):
            self.clinvar_sig = set(_clean_clinsigs([v for v in _tolist(self.clinvar_sig) if v != "nan" and not pd.isna(v)]))
        else:
            self.clinvar_sig = set()

    def init_gnomad_MAF(self):
        """
        It is possible that the MAF is a list of values separated by a semicolon. If so, parse the list and obtain the maximum value.
        """
        maf_values = _tolist(self.gnomad_MAF)
        maf = np.nanmax(pd.to_numeric(maf_values, errors='coerce'))
        if np.isnan(maf):
            maf = 0
        self.gnomad_MAF = maf

    def init_clinvar_sig(self):
        """
        Parse the list of strings, remove invalid values, and assign unique values to the clinvar_sig attribute as a set.
        """
        sigs = set([sig for sig in str(self.clinvar_sig).split(";") if \
                not sig.startswith("CLNSIGCONF") and not sig.startswith("ID") and not sig.startswith("CLNHGVS")])
        self.clinvar_sig = sigs

    def validate_variant_info(self):
        if not hasattr(self,'aa_ref') or not (Variant.is_nan(self.aa_ref) or self.aa_ref in self.valid_alleles.union({"*","-"})):
            raise ValueError(f"aa_ref must be a valid amino acid, '*', '-', or NaN, not {self.aa_ref} for {self.ID}")
        if not hasattr(self,'aa_alt') or not (Variant.is_nan(self.aa_alt) or self.aa_alt in self.valid_alleles.union({"*","-"})):
            raise ValueError(f"aa_alt must be a valid amino acid, '*', or '-', or NaN, not {self.aa_alt} for {self.ID}")
        if not hasattr(self, "HGNC_id"):
            raise ValueError("HGNC_id must be given")
        if not hasattr(self, "aa_pos"):
            self.aa_pos = pd.to_numeric(self.aa_pos, errors='coerce')
            raise ValueError("aa_pos must be given but can be NaN")
    
    @property
    def present_in_gnomAD(self):
        return not pd.isna(self.gnomad_MAF) and self.gnomad_MAF > 0
    
    @property
    def is_pathogenic_or_likely_pathogenic(self) -> bool:
        return len(self.clinvar_sig.intersection({"Pathogenic","Pathogenic/Likely_pathogenic","Likely_pathogenic"})) > 0

    @property
    def is_benign_or_likely_benign(self) -> bool:
        return len(self.clinvar_sig.intersection({"Benign","Benign/Likely_benign","Likely_benign"})) > 0

    @property
    def is_conflicting(self) -> bool:
        return len(self.clinvar_sig.intersection({"Conflicting_classifications_of_pathogenicity",})) > 0

    @property
    def is_uncertain(self) -> bool:
        return len(self.clinvar_sig.intersection({"Uncertain_significance",})) > 0

    @property
    def valid_alleles(self):
        return {'A','C','D','E',
                'F','G','H','I',
                'K','L','M','N',
                'P','Q','R','S',
                'T','V','W','Y'}
        
    @property
    def is_missense(self):
        if not (hasattr(self, "aa_ref") and hasattr(self, "aa_alt") and  hasattr(self, "aa_pos")):
            return False
        return self.aa_ref in self.valid_alleles and \
            self.aa_alt in self.valid_alleles and \
            self.aa_ref != self.aa_alt and \
            self.aa_pos > 0
        
    @staticmethod
    def is_nan(val):
        return isinstance(val, float) and np.isnan(val)

    @property
    def is_synonymous(self):
        return self.aa_ref == self.aa_alt or \
            (self.aa_ref in self.valid_alleles and Variant.is_nan(self.aa_alt))
    
    @property
    def stop_gain(self):
        return self.aa_ref in self.valid_alleles and self.aa_alt == "*"
    
    @property
    def stop_loss(self):
        return self.aa_ref == "*" and self.aa_alt in self.valid_alleles
    
    @property
    def stop_retained(self):
        return (self.aa_ref == "*" and self.aa_alt == "*") or \
            (self.aa_ref == "*" and Variant.is_nan(self.aa_alt) )
    
    @property
    def noncoding(self):
        return self.aa_ref == "-" and pd.isna(self.aa_alt)
    
    @property
    def score(self):
        return self.auth_reported_score
    
      
class AminoAcidVariant(Variant):
    def __eq__(self,other):
        return self.aa_ref == other.aa_ref and \
            self.aa_alt == other.aa_alt and \
            self.aa_pos == other.aa_pos and \
            self.HGNC_id == other.HGNC_id

class NucleotideVariant(Variant):
    def validate_variant_info(self):
        super().validate_variant_info()
        return
        if not hasattr(self, "auth_transcript_id") \
            or not isinstance(self.auth_transcript_id, str) or \
                not len(self.auth_transcript_id):
            raise ValueError("auth_transcript_id must be a string")
        if not hasattr(self, "transcript_pos"):
            raise ValueError("transcript_pos must be provided")
        if not hasattr(self, "transcript_ref") \
            or not isinstance(self.transcript_ref, str) or \
                self.transcript_ref not in self.valid_nucleotides:
            raise ValueError(f"transcript_ref must be a valid nucleotide, not {self.transcript_ref} for {self.ID}")
        if not hasattr(self, "transcript_alt") \
            or not isinstance(self.transcript_alt, str) or \
                self.transcript_alt not in self.valid_nucleotides:
            raise ValueError(f"transcript_alt must be a valid nucleotide, not {self.transcript_alt} for {self.ID}")
        
    @property
    def valid_nucleotides(self):
        return {'A','C','G','T'}
    
    def __eq__(self,other):
        return self.aa_ref == other.aa_ref and \
            self.aa_alt == other.aa_alt and \
            self.aa_pos == other.aa_pos and \
            self.HGNC_id == other.HGNC_id and \
            self.auth_transcript == other.auth_transcript and \
            self.transcript_pos == other.transcript_pos and \
            self.transcript_ref == other.transcript_ref and \
            self.transcript_alt == other.transcript_alt
