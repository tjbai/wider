#!/bin/bash
#SBATCH -A m4789_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH --job-name=open_r1/eval
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
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
