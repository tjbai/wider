#!/bin/bash
#SBATCH --account=m4789_g
#SBATCH -constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=open_r1/train
#SBATCH --gres=gpu:4
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL
#SBATCH --exclusive

if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch slurm/train.slurm [options]"
  echo "Options:"
  echo "  --model MODEL            Model name"
  echo "  --task TASK              Task name (e.g. sft, grpo)"
  echo "  --config SUFFIX          Configuration suffix (e.g. demo, v00.00)"
  echo "  --accelerator CONFIG     Accelerator configuration name (e.g. zero3)"
  echo "  --dp N                   Data parallelism for vLLM server (default: 1)"
  echo "  --tp N                   Tensor parallelism for vLLM server (default: 1)"
  echo "  --args \"ARGS\"          Optional arguments to pass to the training script"
  exit 0
fi

set -x -e
set -euo pipefail

source ~/.bashrc
source .venv/bin/activate
START_TIME=$(date +%s)
echo "START TIME: $(date)"

MODEL="DeepSeek-R1-Distill-Qwen-1.5B"
TASK="grpo"
CONFIG_SUFFIX="sanity_perl"
ACCELERATOR="zero2"
DP=1
TP=1
OPTIONAL_ARGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --config)
      CONFIG_SUFFIX="$2"
      shift 2
      ;;
    --accelerator)
      ACCELERATOR="$2"
      shift 2
      ;;
    --dp)
      DP="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --args)
      OPTIONAL_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" || -z "$TASK" || -z "$CONFIG_SUFFIX" || -z "$ACCELERATOR" ]]; then
  echo "Error: Missing required arguments"
  echo "Run with --help for usage information"
  exit 1
fi

CONFIG_FILE=recipes/$MODEL/$TASK/config_$CONFIG_SUFFIX.yaml
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')

IFS=' ' read -ra ARGS <<< "$OPTIONAL_ARGS"
for arg in "${ARGS[@]}"; do
    if [[ "$arg" == "--gradient_accumulation_steps="* ]]; then
        GRAD_ACC_STEPS="${arg#*=}"
        break
    fi
done

MODEL=$(grep 'model_name_or_path:' $CONFIG_FILE | awk '{print $2}')
REVISION=$(grep 'model_revision:' $CONFIG_FILE | head -n 1 | awk '{print $2}')

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=6000
TRAIN_NODES=("${NODELIST[@]}")

USE_VLLM="false"
if [[ -f "$CONFIG_FILE" ]] && grep -qE '^\s*use_vllm:\s*true' "$CONFIG_FILE"; then
    USE_VLLM="true"
fi
if [[ "$USE_VLLM" == "true" ]]; then
     TRAIN_NODES=("${NODELIST[@]:0:$((NUM_NODES - 1))}")
     VLLM_NODE=${NODELIST[-1]} # Last node
     WORLD_SIZE=$((WORLD_SIZE - GPUS_PER_NODE))
     NUM_NODES=$((NUM_NODES - 1))
     srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE trl vllm-serve --model $MODEL --revision $REVISION --tensor_parallel_size $TP --data_parallel_size $DP &

     OPTIONAL_ARGS="$OPTIONAL_ARGS --vllm_server_host=$VLLM_NODE"
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ABORT_ON_ERROR=1
export NCCL_SOCKET_NRETRY=10
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

export HF_DATASETS_CACHE=/tmp/$USER/hf_dataset_cache
mkdir -p $HF_DATASETS_CACHE

export CMD=" \
    src/open_r1/$TASK.py --config $CONFIG_FILE $OPTIONAL_ARGS
    "

export LAUNCHER="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --max_restarts 1 \
    --tee 3 \
    "

NODELIST=$(IFS=,; echo "${TRAIN_NODES[*]}")

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_NODES \
    --nodelist=$NODELIST
    "

clear
srun $SRUN_ARGS bash -c "$LAUNCHER $CMD" 2>&1

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"

