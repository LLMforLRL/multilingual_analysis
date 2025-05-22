#!/usr/bin/env bash
#SBATCH --job-name=neuron-detect
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out


# once the prompt switches to the compute node:
module load cuda/12.3.2               # or the version on that node
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PLND

python - <<'PY'
import torch, os, subprocess, textwrap
print("Torch              :", torch.__version__)
print("CUDA in torch      :", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("Device count       :", torch.cuda.device_count())
print(subprocess.check_output(["nvidia-smi", "-L"]).decode())
PY
