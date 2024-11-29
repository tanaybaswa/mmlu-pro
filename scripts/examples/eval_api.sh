#!/bin/bash

cd ../../

python evaluate_from_api.py \
                 --model_name jamba-1.5-large \
                 --output_dir jamba_large_eval_results \
                 --assigned_subjects all \
                 --questions_per_topic 26
