import pandas as pd
import numpy as np
from pathlib import Path

class PillarProjectDataframe:
    def __init__(self, data_path: Path|str):
        self.data_path = Path(data_path)
        self.init_data()

    def init_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")
        self.dataframe = pd.read_csv(self.data_path)

class FunctionalDataset:
    def __init__(self, dataframe: pd.DataFrame):
        self._init_dataframe(dataframe)

    def _init_dataframe(self, dataframe : pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if len(dataframe.Dataset.unique()) != 1:
            raise ValueError("dataframe must contain only one dataset")
        if not len(dataframe):
            raise ValueError("dataframe must contain at least one row")
        self.dataframe = dataframe
        self.variants = [self._init_variant(row) for _, row in dataframe.iterrows()]
        self._init_variant_sample_assignments()

    def _init_variant(self, row):
        if row.nucleotide_or_aa == "nucleotide":
            return NucleotideVariant(row)
        elif row.nucleotide_or_aa == "aa":
            return AminoAcidVariant(row)
        else:
            raise ValueError("nucleotide_or_aa must be either 'nucleotide' or 'aa'")

    def __len__(self):
        return len(self.dataframe)

    def _init_variant_sample_assignments(self):
        NSamples = 4 # P/LP, B/LB, gnomAD, synonymous
        self._sample_assignments = np.zeros((len(self),NSamples), dtype=bool)
        for i,variant in enumerate(self.variants):
            if variant.is_synonymous:
                self._sample_assignments[i,3] = True
                continue
            if variant.is_missense:
                if variant.present_in_gnomAD:
                    self._sample_assignments[i,2] = True
                if variant.is_pathogenic_or_likely_pathogenic:
                    self._sample_assignments[i,0] = True
                if variant.is_benign_or_likely_benign:
                    self._sample_assignments[i,1] = True
    @property
    def sample_assignments(self):
        return self._sample_assignments
    
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
            if self.sample_assignments[:,sample_index].sum() > 0:
                yield self.scores[self.sample_assignments[:,sample_index]],sample_names[sample_index]
    
    @property
    def scores(self):
        return np.array([variant.score for variant in self.variants])
    
    def __getitem__(self, idx):
        return self.variants[idx]
    
class Variant:
    def __init__(self, variant_info: pd.Series):
        self._init_variant_info(variant_info)
    
    def _init_variant_info(self, variant_info: pd.Series):
        for k, v in variant_info.items():
            setattr(self, k, v)
        self.validate_variant_info()

    def validate_variant_info(self):
        if not hasattr(self,'aa_ref') or self.aa_ref not in self.valid_alleles.union({"*","-"}):
            raise ValueError("aa_ref must be a valid amino acid, '*', or '-'")
        if not hasattr(self, "aa_alt") or \
            (self.aa_alt not in self.valid_alleles.union({"*",'-',pd.NA})):
            raise ValueError("aa_alt must be a valid amino acid, '*', or '-', or NaN")
        if not hasattr(self, "HGNC_id"):
            raise ValueError("HGNC_id must be given")
        if not hasattr(self, "aa_pos"):
            raise ValueError("aa_pos must be given but can be NaN")
    
    @property
    def present_in_gnomAD(self):
        return not pd.isna(self.gnomad_MAF) and self.gnomad_MAF > 0
    
    def clinvar_sig_missing(self):
        if not hasattr(self, "clinvar_sig") or self.is_nan(self.clinvar_sig):
            return True
        return False
    @property
    def is_pathogenic_or_likely_pathogenic(self):
        if self.clinvar_sig_missing():
            return False
        sig = self.clinvar_sig.split(";")[0]
        return sig in {"Pathogenic","Pathogenic/Likely_pathogenic","Likely_pathogenic"}

    @property
    def is_benign_or_likely_benign(self):
        if self.clinvar_sig_missing():
            return False
        sig = self.clinvar_sig.split(";")[0]
        return sig in {"Benign","Benign/Likely_benign","Likely_benign"}

    @property
    def is_conflicting(self):
        if self.clinvar_sig_missing():
            return False
        sig = self.clinvar_sig.split(";")[0]
        return sig == "Conflicting_classifications_of_pathogenicity"

    @property
    def is_uncertain(self):
        if self.clinvar_sig_missing():
            return False
        sig = self.clinvar_sig.split(";")[0]
        return sig == "Uncertain_significance"

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
        if not hasattr(self, "auth_transcript_id") \
            or not isinstance(self.auth_transcript_id, str) or \
                not len(self.auth_transcript_id):
            raise ValueError("auth_transcript_id must be a string")
        if not hasattr(self, "transcript_pos"):
            raise ValueError("transcript_pos must be provided")
        if not hasattr(self, "transcript_ref") \
            or not isinstance(self.transcript_ref, str) or \
                self.transcript_ref not in self.valid_nucleotides:
            raise ValueError("transcript_ref must be a valid nucleotide")
        if not hasattr(self, "transcript_alt") \
            or not isinstance(self.transcript_alt, str) or \
                self.transcript_alt not in self.valid_nucleotides:
            raise ValueError("transcript_alt must be a valid nucleotide")
        
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
