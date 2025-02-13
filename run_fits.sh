#!/bin/bash
#SBATCH --job-name=run_fits
#SBATCH --output=/work/pedjas_lab/zeiberg.d/pillar_project/logs/run_fits_%j_%A_%a.log
#SBATCH --error=/work/pedjas_lab/zeiberg.d/pillar_project/logs/run_fits_%j_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH --partition=standard
#SBATCH --array=1-54
USER=$(whoami)
source activate /work/pedjas_lab/zeiberg.d/pillar_project/env/

DATASET=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /work/pedjas_lab/zeiberg.d/pillar_project/data/dataset_names.txt)
SAVEDIR=/work/pedjas_lab/zeiberg.d/pillar_project/fit_results/
mkdir -p $SAVEDIR
DATAFRAME=/work/pedjas_lab/zeiberg.d/pillar_project/data/pillar_data_condensed_01_28_25.csv
echo $DATASET
for i in {1..1000}
do
    echo "Iteration $i"
    python /work/pedjas_lab/zeiberg.d/pillar_project/src/pillar_project/run_fits.py run_single_fit $DATASET $DATAFRAME $SAVEDIR/fit_results_${DATASET}_${i}/
done

