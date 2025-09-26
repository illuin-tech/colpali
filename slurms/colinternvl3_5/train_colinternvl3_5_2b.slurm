#!/bin/bash
#SBATCH --output=slurm_logs/%x_%j.log
#SBATCH --error=slurm_logs/%x_%j.log
#SBATCH -A qjm@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --job-name=train-colinternvl3_5-2b
# #SBATCH --cpu-bind=none
#SBATCH --mem-bind="local"
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

module purge
module load arch/h100 cuda/12.8.0

export HF_DATASETS_CACHE=$SCRATCH/datasets
export HF_HOME=$SCRATCH/.cache/huggingface
export UV_CACHE_DIR=$SCRATCH
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_PROJECT="train-colinternvl3_5-2b"
export WANDB_NAME="train-colinternvl3_5-2b"
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export HF_HUB_OFFLINE=1

source .venv/bin/activate
wandb offline
echo "launching training script"
accelerate launch --config_file scripts/configs/accelerate_configs/single_node_config.yml scripts/configs/internvl/train_colinternvl3_5_2b_model.py
cd ../../mteb
source .venv/bin/activate
python -m experiments.colinternvl.eval_colinternvl --model_name models/colinternvl3_5_2b
