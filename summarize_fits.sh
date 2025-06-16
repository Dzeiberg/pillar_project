#!/bin/bash

DATASETS_FILEPATH=files/AllDatasets.txt
DISCOVERY_FITS_DIR=files/model_selection_04_03_2025_P3/
SCORESETS_DIR=files/dataframe/final_scoresets
SUMMARIES_DIR=files/model_selection_05_16_25_MedianPrior_finalQuantile_05/

if [ ! -f "$DATASETS_FILEPATH" ]; then
    echo "File $DATASETS_FILEPATH not found!"
    exit 1
fi

FINAL_QUANTILE=0.05

while IFS= read -r dataset_name; do
    echo $dataset_name
    python -m model_selection.model_selection_results summarize_dataset_fits \
    $DISCOVERY_FITS_DIR \
    $dataset_name \
    $SCORESETS_DIR \
    $SUMMARIES_DIR \
    --final_quantile=$FINAL_QUANTILE \
    --best_n_components \
    --n_fits_to_load=1000
done < "$DATASETS_FILEPATH"