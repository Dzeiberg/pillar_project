cat /home/dzeiberg/pillar_project/dataset_names.txt | while read dataset_name
do
    echo $dataset_name
    python -m model_selection.model_selection_results summarize_dataset_fits \
    /data/dzeiberg/pillar_project/fit_results/ \
    $dataset_name \
    "/data/dzeiberg/pillar_project/dataframe/pillar_data_condensed_01_28_25.csv" \
    "/data/dzeiberg/pillar_project/fit_summaries/" \
    --min_exceeding=0.0 --final_quantile=0.5
done