#!/bin/bash

# Simple helper script to train MDLT using the BoDA algorithm.
#
# Usage:
#   ./run_boda.sh [DATASET] [OUTPUT_NAME] [DATA_DIR] [OUTPUT_DIR]
#
# Arguments:
#   DATASET      Dataset name (default: PACS)
#   OUTPUT_NAME  Experiment folder name (default: boda_experiment)
#   DATA_DIR     Path to dataset root (default: ./data)
#   OUTPUT_DIR   Directory to store outputs (default: ./output)

DATASET=${1:-PACS}
OUTPUT_NAME=${2:-boda_experiment}
DATA_DIR=${3:-./data}
OUTPUT_DIR=${4:-./output}

python -m mdlt.train \
  --dataset "$DATASET" \
  --algorithm BoDA \
  --output_folder_name "$OUTPUT_NAME" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --hparams '{"resnet18": true}'
