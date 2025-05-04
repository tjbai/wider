sbatch slurm/train_perl-4xA100.slurm \
    --model DeepSeek-R1-Distill-Qwen-1.5B \
    --config sanity \
    --accelerator zero2 \
    --task grpo \
