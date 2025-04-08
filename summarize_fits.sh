#!/bin/bash

DATASETS_FILEPATH=files/dataframe/Datasets_03262025.csv
DISCOVERY_FITS_DIR=files/model_selection_04_03_2025_P3/
SCORESETS_DIR=files/dataframe/final_scoresets
SUMMARIES_DIR=files/model_selection_04_03_2025_P3_summaries/

FINAL_QUANTILE=.05
# ------------------- FOR FINAL SUMMARIZING --------------------
MIN_EXCEEDING=0.95
# ------------------- FOR MODEL SELECTION -------------------
# MIN_EXCEEDING=.5

# for dataset_name in "ASPA_Grønbæk-Thygesen_2024_abundance" "ASPA_Grønbæk-Thygesen_2024_toxicity" "PTEN_Mighell_2018" "LARGE1_Ma_2024" "FKRP_Ma_2024" "BRCA2_Erwood_2022" "BRCA2_Sahu_2025_HDR" "BRCA2_Huang_2025_HDR" "RAD51C_Olvera-León_2024_z_score_D4_D14" "NPC1_Erwood_2022_HEK293T" "DDX3X_Radford_2023_cLFC_day15" "BAP1_Waters_2024" "JAG1_Gilbert_2024" "VHL_Buckley_2024" "BRCA1_Findlay_2018" "TPK1_Weile_2017" "TP53_Kato_2003_h1433snWT" "TP53_Kato_2003_WAF1nWT" "TP53_Kato_2003_P53R2nWT" "TP53_Kato_2003_NOXAnWT" "BRCA1_Adamovich_2022_Cisplatin" "TP53_Kato_2003_MDM2nWT" "TP53_Kato_2003_GADD45nWT" "TP53_Kato_2003_BAXnWT" "TP53_Kato_2003_AIP1nWT" "TP53_Giacomelli_2018_p53null_etoposide" "BRCA1_Adamovich_2022_HDR" "TP53_Giacomelli_2018_p53null_Nutlin3" "TP53_Giacomelli_2018_p53WT_Nutlin3" "TP53_Giacomelli_2018_combined_score" "TP53_Fortuno_2021_Kato_meta" "TP53_Fayer_2021_meta" "TP53_Boettcher_2019" "SCN5A_Ma_2024_current_density" "PTEN_Matreyek_2018" "OTC_Lo_2023" "NDUFAF6_Sung_2024" "MSH2_Jia_2021" "CHK2_Gebbia_2024" "KCNQ4_Zheng_2022_v12_homozygous" "KCNQ4_Zheng_2022_current_homozygous" "CRX_Shepherdson_2024" "F9_Popp_2025_carboxy_F9_specific"
for dataset_name in "KCNE1_Muhammad_2024_potassium_flux" \
"KCNH2_O_Neill_2024_surface_expression" \
"F9_Popp_2025_carboxy_gla_motif" \
"KCNH2_Jiang_2022" \
"KCNE1_Muhammad_2024_presence_of_WT"
do
    echo $dataset_name
    python -m model_selection.model_selection_results summarize_dataset_fits \
    $DISCOVERY_FITS_DIR \
    $dataset_name \
    $SCORESETS_DIR \
    $SUMMARIES_DIR \
    --min_exceeding=$MIN_EXCEEDING \
    --final_quantile=$FINAL_QUANTILE \
    --best_n_components \
    --n_fits_to_load=1000
done
