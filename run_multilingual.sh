#!/bin/bash

# Example script to run inference on multiple multilingual datasets
# This script demonstrates how to use the modular run.py with different datasets, languages, and models
# Usage: ./run_multilingual.sh [--test]

# Check if test mode is enabled
TEST_MODE=""
for arg in "$@"; do
  if [ "$arg" == "--test" ]; then
    TEST_MODE="--test_mode"
    echo "Running in TEST MODE - will only process a few examples from each dataset"
  fi
done

# Define models to use
MODELS=(
  # Using VLLM
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

  'Skywork/Skywork-OR1-7B'
  'Skywork/Skywork-OR1-32B'
)

# Define query languages for test
LANGUAGES=(
  "EN"
  "FR"
  "DE"
  "ZH"
  "JA"
  "RU"
  "ES"
  "SW"
  "BN"
  "TE"
  "TH"
)

# Define languages for thinking
LANGUAGES_THINK=(
  # "default" # Default language for thinking, always same as query language
  "EN"
  "FR"
  "DE"
  "ZH"
  "JA"
  "RU"
  "ES"
  "SW"
  "BN"
  "TE"
  "TH"
)

# Define datasets to use
DATASETS=(
  # # AIME combined dataset
  "aime_combined:problem:answer"

  # # GPQA dataset
  "shanchen/gpqa_diamond_mc_multilingual:problem:solution"
  
  # # MGSM dataset
  "juletxara/mgsm:question:answer_number:test"

  # # (Not used) AIW hard multilingual dataset
  # "shanchen/aiw_hard_multilingual:problem:answer"
)

# Maximum number of parallel processes to run
MAX_PARALLEL=1 # Set to 1 for vllm runnings

# Create a temporary file to track running processes
PIDFILE=$(mktemp)
echo "" > "$PIDFILE"

# Function to clean up the PID file on exit
cleanup() {
  rm -f "$PIDFILE"
}
trap cleanup EXIT

# Function to run a job and maintain the job count
run_job() {
  # Parse dataset info
  IFS=':' read -r DATASET QUESTION_FIELD ANSWER_FIELD SPLIT <<< "$1"
  MODEL="$2"
  LANG="$3"
  LANG_THINK="$4"

  echo "Starting inference with:"
  echo "  Model: $MODEL"
  echo "  Language: $LANG"
  echo "  Language Think: $LANG_THINK"
  echo "  Dataset: $DATASET"
  echo "  Fields: $QUESTION_FIELD / $ANSWER_FIELD"
  echo "  Split: $SPLIT"
  
  # Set max new tokens
  MAX_TOKENS=16834
  
  # Set seed for reproducibility
  SEED=2025
  # SEED=0 # Set 0 for forcing greedy decoding

  # Create a unique log file for this run
  LOG_FILE="logs/${MODEL//\//_}_${DATASET//\//_}_${LANG}_think_${LANG_THINK}.log"
  mkdir -p logs
  
  # Run the inference
  {
    # Convert LANG to lowercase for config
    LANG_LOWER=$(echo "$LANG" | tr '[:upper:]' '[:lower:]')
    
    # For these datasets, we need to pass the language as the split parameter
    # sbatch --job-name=gpu_job_xlarge \
    # --partition=bch-gpu-xlarge --account=bch --gres=gpu:xlarge:4 --mem=256GB \
    # --time=8:00:00 --output=logs/$MODEL/$DATASET/$LANG\_think\_$LANG_THINK.%j.out \
    # --wrap="conda run -n s2 \
    python run.py \
      --mname "$MODEL" \
      --lang "$LANG" \
      --lang_think "$LANG_THINK" \
      --dataset "${DATASET}" \
      --question_field "$QUESTION_FIELD" \
      --answer_field "$ANSWER_FIELD" \
      --split "${LANG_LOWER}" \
      --max_tokens "$MAX_TOKENS" \
      --cache_dir "/temp_work/ch225816/hf" \
      --seed "$SEED" \
      $TEST_MODE \
      # "

    echo "Completed inference for $DATASET with $MODEL in $LANG"
    echo "----------------------------------------"
  } > "$LOG_FILE" 2>&1 &
  
  # # Get the PID of the background process
  # local pid=$!
  
  # # Add the PID to our tracking file
  # echo "$pid" >> "$PIDFILE"
  
  # # Print the PID of the background process
  # echo "Started job with PID $pid (log: $LOG_FILE)"
}

# Queue up all jobs
echo "Queueing inference jobs..."
for LANG_THINK in "${LANGUAGES_THINK[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for LANG in "${LANGUAGES[@]}"; do
      for DATASET_INFO in "${DATASETS[@]}"; do
        # Parse dataset info
        IFS=':' read -r DATASET QUESTION_FIELD ANSWER_FIELD SPLIT <<< "$DATASET_INFO"
        
        # For datasets that need language config, we'll pass the language as the split parameter
        if [[ "$DATASET" == "juletxara/mgsm" || "$DATASET" == "shanchen/aiw_hard_multilingual" || "$DATASET" == "shanchen/aime_2024_multilingual" || "$DATASET" == "shanchen/aime_2025_multilingual" || "$DATASET" == "aime_combined" || "$DATASET" == "shanchen/gpqa_diamond_mc_multilingual" ]]; then
          # Convert LANG to lowercase for config
          LANG_LOWER=$(echo "$LANG" | tr '[:upper:]' '[:lower:]')
          # Use the lowercase language as the split parameter
          SPLIT="${LANG_LOWER}"
        fi
        
        # Reconstruct the dataset info
        DATASET_INFO="${DATASET}:${QUESTION_FIELD}:${ANSWER_FIELD}:${SPLIT}"
        
        run_job "$DATASET_INFO" "$MODEL" "$LANG" "$LANG_THINK"
      done
    done
  done
done

# # Wait for all background jobs to finish
# echo "All jobs queued. Waiting for completion..."
# wait

echo "All inference runs completed!"
echo "Check the logs directory for individual job logs."
