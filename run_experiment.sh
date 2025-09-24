#!/bin/bash
#SBATCH --time=0:40:00
#SBATCH --mem=20GB
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=eb4727@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=slurm/%j.out

# Load conda environment
module purge
source /vast/eb4727/miniconda3/etc/profile.d/conda.sh
conda activate noise_generalization

# Read command-line arguments
lr=$1
corrupt_frac=$2
model_name=$3

echo "Running experiment with:"
echo "Learning rate: $lr"
echo "Corruption fraction: $corrupt_frac"
echo "Model: $model_name"

python train.py \
  --peak_lr="$lr" \
  --n_epochs=200 \
  --corrupt_frac="$corrupt_frac" \
  --model_name="$model_name" \
  --wandb_mode='online'
