#!/bin/bash
#BSUB -P acc_pejaverlab
#BSUB -q express
#BSUB -W 3:59
#BSUB -J pillar_project[1-10]
#BSUB -n 10
#BSUB -R span[hosts=1]
#BSUB -o /sc/arion/projects/pejaverlab/users/zeibed01/pillar_project/logs/pillar_project_log.out.%I
source /hpc/users/zeibed01/.bashrc
source activate /hpc/users/zeibed01/.conda/envs/pillar_project

# set save directory
SAVEDIR=/sc/arion/projects/pejaverlab/users/zeibed01/pillar_project/results/
mkdir -p $SAVEDIR
# set dataframe filepath
DATAFRAME=/sc/arion/projects/pejaverlab/users/zeibed01/pillar_project/data/pillar_data_condensed_01_28_25.csv

NUM_DATASETS=54
for dataset_num in $(seq 1 $NUM_DATASETS)
do
        DATASET=$(sed -n "${dataset_num}p" /sc/arion/projects/pejaverlab/users/zeibed01/pillar_project/data/dataset_names.txt)
        echo "Running $DATASET - iteration $LSB_JOBINDEX";
        python -u /sc/arion/projects/pejaverlab/users/zeibed01/pillar_project/run_fits.py run_single_fit $DATASET $DATAFRAME $SAVEDIR/fit_results_${DATASET}_${LSB_JOBINDEX}/ --num_fits 5 --core_limit 10
done