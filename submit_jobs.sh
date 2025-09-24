#!/bin/bash

corrupt_fracs=(
  0
  0.1
  0.2
  0.3
  0.4
  0.5
)

model_names=(
  # resnet20
  # resnet32
  # resnet44
  # resnet56
  # resnet110
  resnet80
)

learning_rates=(
  # 0.1
  # 0.01
  0.001
  # 0.0001
)

for corrupt_frac in "${corrupt_fracs[@]}"; do
  for model_name in "${model_names[@]}"; do
    for lr in "${learning_rates[@]}"; do
      echo "Submitting job: lr=$lr, corrupt_frac=$corrupt_frac, model=$model_name"
      sbatch run_experiment.sh "$lr" "$corrupt_frac" "$model_name"
    done
  done
done
