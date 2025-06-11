#!/bin/bash
# run_blip_pipeline.sh
# End-to-end BLIP pipeline: preprocess, train, and inference with PHI removal
# Usage: bash run_blip_pipeline.sh <input_csv> <image_dir> <output_dir> <test_image> <question>

set -e
INPUT_CSV="$1"
IMAGE_DIR="$2"
OUTPUT_DIR="$3"
TEST_IMAGE="$4"
QUESTION="$5"

PREPROCESSED_CSV="${OUTPUT_DIR}/preprocessed.csv"

# Step 1: Preprocess (de-identify) data
python3 src/blip/preprocess.py --input_csv "$INPUT_CSV" --output_csv "$PREPROCESSED_CSV"

# Step 2: Train BLIP model
python3 src/blip/train_blip.py --data_path "$PREPROCESSED_CSV" --image_dir "$IMAGE_DIR" --output_dir "$OUTPUT_DIR"

# Step 3: Inference with PHI removal
python3 src/blip/infer_blip.py --model_path "$OUTPUT_DIR" --image_path "$TEST_IMAGE" --question "$QUESTION"
