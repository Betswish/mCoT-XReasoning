#!/bin/bash
# SLURM batch script for running a PyTorch job with 4 xlarge GPUs

#SBATCH --job-name=ja500MATH0_30_8lora_nokl
#SBATCH --account=bch                      # Account to be charged
#SBATCH --partition=bch-gpu-xlarge
#SBATCH --gres=gpu:xlarge:4               # Request 4 xlarge GPUs
#SBATCH --mem=100GB                        # Memory allocation
#SBATCH --time=99:00:00                    # Max runtime
#SBATCH --mail-type=END,FAIL               # Email notifications
#SBATCH --mail-user=shan.chen@childrens.harvard.edu
#SBATCH --output=logs/slurm-%j.out         # Standard output log

# Activate your environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate myrl
export HF_HOME=/temp_work/ch225816/hf


# Run your Python script
#accelerate launch --config_file review/accelerate_configs/deepspeed_zero3.yaml review/qwen_sft.py
#accelerate launch --config_file review/accelerate_configs/deepspeed_zero3.yaml review/qwen7_sft.py

cd /home/ch225816/grpo_attempts

Generate unique ports based on Slurm job ID to avoid conflicts
RAY_PORT=$((6379 + ($SLURM_JOB_ID % 1000)))
DASHBOARD_PORT=$((8265 + ($SLURM_JOB_ID % 1000)))

# Check if our specific ports are available
if netstat -tuln | grep -q ":$RAY_PORT " || netstat -tuln | grep -q ":$DASHBOARD_PORT "; then
    echo "Warning: Our assigned ports $RAY_PORT or $DASHBOARD_PORT are already in use!"
    echo "This might indicate a port collision. Trying to find alternative ports..."
    
    # Try a few alternative ports
    for i in {1..5}; do
        ALT_RAY_PORT=$((RAY_PORT + i * 100))
        ALT_DASHBOARD_PORT=$((DASHBOARD_PORT + i * 100))
        
        if ! netstat -tuln | grep -q ":$ALT_RAY_PORT " && ! netstat -tuln | grep -q ":$ALT_DASHBOARD_PORT "; then
            echo "Using alternative ports: Ray $ALT_RAY_PORT, Dashboard $ALT_DASHBOARD_PORT"
            RAY_PORT=$ALT_RAY_PORT
            DASHBOARD_PORT=$ALT_DASHBOARD_PORT
            break
        fi
    done
else
    echo "Our ports are free, starting fresh Ray cluster..."
fi

echo "Starting Ray cluster (head node, port $RAY_PORT, dashboard port $DASHBOARD_PORT)..."
ray start --head --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 --dashboard-port=$DASHBOARD_PORT \
    --include-dashboard=true \
    --disable-usage-stats


# # # Start vLLM server on GPUs 0,1 in background
# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model shanchen/ds-limo-ja-500 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml trl_hf_es.py

# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model shanchen/ds-limo-fr-250 --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml trl_grpo_fr.py

CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model shanchen/ds-limo-ja-500 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml trl_grpo.py

# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --tensor_parallel_size 1 &
# CUDA_VISIBLE_DEVICES=1 accelerate launch trl_grpo.py 

# Alternative configs (uncomment to try):
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config_file accelerate_configs/fsdp2.yaml trl_grpo.py
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config_file accelerate_configs/deepspeed_zero1.yaml trl_grpo.py 
#deepseek-ai/DeepSeek-R1-0528-Qwen3-8B







