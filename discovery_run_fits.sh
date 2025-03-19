#!/bin/bash
#SBATCH --job-name=job_array_example
#SBATCH --output=/scratch/zeiberg.d/pillar_project/logs/output_%A_%a.out
#SBATCH --error=/scratch/zeiberg.d/pillar_project/logs/error_%A_%a.err
#SBATCH --array=1-50          # Adjust the range as needed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    # Request 32 cores per job
#SBATCH --time=23:59:00       # Adjust the time limit as needed
#SBATCH --partition=short  # Adjust the partition as needed

# Load necessary modules
module load miniconda3/23.5.2
source activate pillar_project

# Define the task for each array job
echo "Starting task $SLURM_ARRAY_TASK_ID on $(hostname)"
##########################################################

SAVEDIR=/work/talisman/dzeiberg/pillar_project/model_selection_fits_03182025/
mkdir -p $SAVEDIR
SCORESETS_DIR=/work/talisman/dzeiberg/pillar_project/processed_scoresets/
SET_ID="Set_1"
datasets_file="${SET_ID}_Datasets.txt"
NDatasets=$(wc -l "${datasets_file}" | awk '{print $1}')

for i in {1..20}
do
    echo "Iteration $i"
    for (( SLURM_ARRAY_TASK_ID=1; SLURM_ARRAY_TASK_ID<=$NDatasets; SLURM_ARRAY_TASK_ID++ ))
    do
        echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
        DATASET=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${datasets_file}")
        # echo "Running fit for $DATASET"
        echo "$DATASET ${SCORESETS_DIR}/${SET_ID}_${DATASET}.pkl ${SAVEDIR}/${SET_ID}_${DATASET}_${i}/"
        python -u run_fits.py run_single_fit \
        "${SET_ID}_${DATASET}_${SLURM_ARRAY_TASK_ID}_${i}" \
        ${SCORESETS_DIR}/${SET_ID}_${DATASET}.pkl \
        ${SAVEDIR} \
        --num_fits 100 \
        --core_limit 32
    done
done
