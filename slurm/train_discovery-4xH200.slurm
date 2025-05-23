#!/bin/bash
#SBATCH --account=eisnerlab-hpc
#SBATCH --partition=azgpuh200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h200:4
#SBATCH --job-name=open_r1/train
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --mail-user=tbai4@jh.edu
#SBATCH --mail-type=END,FAIL

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
nvidia-smi

source ~/.bashrc
source .venv/bin/activate
START_TIME=$(date +%s)
echo "START TIME: $(date)"

MODEL="DeepSeek-R1-Distill-Qwen-1.5B"
TASK="grpo"
CONFIG_SUFFIX="sanity_discovery"
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

export ACCELERATE_LOG_LEVEL=debug
export TRANSFORMERS_VERBOSITY=debug
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# export HF_DATASETS_CACHE=/tmp/$USER/hf_dataset_cache
# mkdir -p $HF_DATASETS_CACHE

export CMD=" \
    src/open_r1/$TASK.py --config $CONFIG_FILE $OPTIONAL_ARGS
    "

export LAUNCHER="accelerate launch \
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
    --gpus=4 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 2>&1

