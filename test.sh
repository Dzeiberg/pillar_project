#!/bin/bash
#SBATCH --job-name=test_fits
#SBATCH --output=/work/pedjas_lab/zeiberg.d/pillar_project/logs/test_fit_%j_%A_%a.log
#SBATCH --error=/work/pedjas_lab/zeiberg.d/pillar_project/logs/test_fit_%j_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=00:19:59
#SBATCH --partition=express
#SBATCH --array=1-4
USER=$(whoami)
source activate /work/pedjas_lab/zeiberg.d/pillar_project/env/

DATASET=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /work/pedjas_lab/zeiberg.d/pillar_project/data/dataset_names.txt)
SAVEDIR=/work/pedjas_lab/zeiberg.d/pillar_project/fit_results/
mkdir -p $SAVEDIR
DATAFRAME=/work/pedjas_lab/zeiberg.d/pillar_project/data/pillar_data_condensed_01_28_25.csv
echo $DATASET
for i in {1..2}
do
    echo "Iteration $i"
    python /work/pedjas_lab/zeiberg.d/pillar_project/src/pillar_project/run_fits.py run_single_fit $DATASET $DATAFRAME $SAVEDIR/fit_results_${DATASET}_${i}/ --num_fits 5
done

