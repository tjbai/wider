#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --output=./logs/eval_%x-%j.out
#SBATCH --error=./logs/eval_%x-%j.err
#SBATCH --requeue

set -e

source .venv/bin/activate

TASK_NAME=$1                # e.g. "aime24"
TASKS=$2                    # e.g. "aime24 math_500 amc23"
MODEL_ID=$3                 # e.g. "facebook/opt-6.7b"

OUTPUT_DIR="$SCRATCH/eval_results/${MODEL_ID//\//_}/${TASK_NAME}"
mkdir -p "$OUTPUT_DIR"

uv run accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  run_evals.py \
    --model_name_or_path "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --tasks $TASKS \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95

echo "Done. Results in $OUTPUT_DIR"
