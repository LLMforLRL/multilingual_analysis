#!/usr/bin/env bash
#SBATCH --job-name=neuron-detect
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=ampere&80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out

# --- modules ----------------------------------------------------------
module load cuda/12.3.2
# --- Conda ------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PLND

echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS//,/}

# --- Work ----------------------------------------------------------------
set -e
cd "$SLURM_SUBMIT_DIR"

DATA_DIR=corpus_nllb
FILE_PREFIX=train_100k

for LANG in am ee en fr ha ig rw ln lg om sn st sw tw wo xh yo zu; do
    FILE="$DATA_DIR/${FILE_PREFIX}.${LANG}"
    if [[ -f "$FILE" ]]; then
        echo "=== ${LANG} ==="
        python neuron_detection_gemma3.py "${LANG}" 30000
    else
        echo "--- ${LANG}: corpus missing ($FILE), skipped."
    fi
done
