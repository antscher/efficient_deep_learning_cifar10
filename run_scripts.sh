#!/bin/bash

# Script to run all the specified Python scripts sequentially

echo "Running TD2_wandb_mixup_cos.py"
python3 /users/local/efficient_deep_learning/TD2_wandb_mixup_cos.py

echo "Running TD3_part1_quantization.py"
python3 /users/local/efficient_deep_learning/TD3_part1_quantization.py

echo "Running TD6_distillation.py"
python3 /users/local/efficient_deep_learning/TD6_distillation.py