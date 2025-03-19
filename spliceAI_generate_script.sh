#!/bin/bash

PILLAR_PROJECT_VCF=/data/dzeiberg/pillar_project/dataframe/pillar_data_clinvar38_annotated_final_v13_expanded_031425.vcf.gz
SPLICE_AI_SNV_FILE=/data/dbs/spliceAI/genome_scores_v1.3_ds.20a701bc58ab45b59de2576db79ac8d0/spliceai_scores.masked.snv.hg38.vcf.gz
SPLICE_AI_INDEL_FILE=/data/dbs/spliceAI/genome_scores_v1.3_ds.20a701bc58ab45b59de2576db79ac8d0/spliceai_scores.masked.indel.hg38.vcf.gz

SNV_INTERMEDIATE_FILE=/data/dzeiberg/pillar_project/dataframe/pillar_data_X_SpliceAI_SNV_INTERMEDIATE.vcf.gz
SNV_OUTPUT_FILE=/data/dzeiberg/pillar_project/dataframe/pillar_data_clinvar38_annotated_final_v13_expanded_031425_X_SpliceAI_SNV.vcf.gz

INDEL_INTERMEDIATE_FILE=/data/dzeiberg/pillar_project/dataframe/pillar_data_X_SpliceAI_INDEL_INTERMEDIATE.vcf.gz
INDEL_OUTPUT_FILE=/data/dzeiberg/pillar_project/dataframe/pillar_data_clinvar38_annotated_final_v13_expanded_031425_X_SpliceAI_INDEL.vcf.gz

# SNV
# 1) Generate the intersection of the pillar project vcf file and the spliceAI snv file, using the ID field of the pillar project vcf file
bcftools isec -n=2 -w1 -Oz -o $SNV_INTERMEDIATE_FILE $PILLAR_PROJECT_VCF $SPLICE_AI_SNV_FILE;
# 2) Index the intermediate file
tabix $SNV_INTERMEDIATE_FILE;
# 3) Generate the final output file by annotating the intermediate file with the spliceAI snv file INFO field
bcftools annotate -a $SPLICE_AI_SNV_FILE -c INFO -Oz -o $SNV_OUTPUT_FILE $SNV_INTERMEDIATE_FILE;
# 4) Remove the intermediate file if successful
if [ -f "$SNV_OUTPUT_FILE" ]; then
    rm -f "$SNV_INTERMEDIATE_FILE"
fi

# INDEL
# 1) Generate the intersection of the pillar project vcf file and the spliceAI indel file, using the ID field of the pillar project vcf file
bcftools isec -n=2 -w1 -Oz -o $INDEL_INTERMEDIATE_FILE $PILLAR_PROJECT_VCF $SPLICE_AI_INDEL_FILE;
# 2) Index the intermediate file
tabix $INDEL_INTERMEDIATE_FILE;
# 3) Generate the final output file by annotating the intermediate file with the spliceAI indel file INFO field
bcftools annotate -a $SPLICE_AI_INDEL_FILE -c INFO -Oz -o $INDEL_OUTPUT_FILE $INDEL_INTERMEDIATE_FILE;
# 4) Remove the intermediate file if successful
if [ -f "$INDEL_OUTPUT_FILE" ]; then
    rm -f "$INDEL_INTERMEDIATE_FILE"
fi