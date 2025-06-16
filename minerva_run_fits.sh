#!/bin/bash
#BSUB -P acc_pejaverlab
#BSUB -q premium
#BSUB -W 143:59
#BSUB -n 20
#BSUB -R span[hosts=1]
#BSUB -J pillar_project[1-100]
#BSUB -o /sc/arion/projects/pejaverlab/users/zeibed01/pillar_project/logs/pillar_project_log.out.%I
source /hpc/users/zeibed01/.bashrc
source activate /hpc/users/zeibed01/.conda/envs/pillar_project

# set save directory
SAVEDIR=/sc/arion/projects/pejaverlab/users/zeibed01/pillar_project/model_selection_fits_03262025/
mkdir -p $SAVEDIR
SCORESETS_DIR=/sc/arion/projects/pejaverlab/users/zeibed01/pillar_project/final_scoresets/
datasets_file="Datasets_03262025.csv"
NDatasets=$(wc -l "${datasets_file}" | awk '{print $1}')

for i in {1..100}
do
    echo "Iteration $i"
    for (( SLURM_ARRAY_TASK_ID=1; SLURM_ARRAY_TASK_ID<=$NDatasets; SLURM_ARRAY_TASK_ID++ ))
    do
        echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
        DATASET=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${datasets_file}")
        # echo "Running fit for $DATASET"
        echo "$DATASET ${SCORESETS_DIR}/${DATASET}.pkl ${SAVEDIR}/${DATASET}_${i}/"
        python -u run_fits.py run_single_fit \
        "${DATASET}" \
        ${SCORESETS_DIR}/${DATASET}.pkl \
        ${SAVEDIR} \
        --num_fits 100 \
        --core_limit 20 \
        --run_index $i \
        --component_range [2,3] \
    done
done
