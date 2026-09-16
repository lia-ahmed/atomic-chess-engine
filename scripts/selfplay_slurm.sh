#!/bin/bash
#SBATCH --job-name=atomic-selfplay
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --array=0-7

set -euo pipefail

module load cuda
source myvenv/bin/activate

python src/self_play.py \
  --net ckpts/phase4/iter_001/best.pt \
  --action-map-path action_map.json \
  --out-dir /scratch/$USER/atomic_selfplay/iter_001 \
  --games 100 \
  --simulations 400 \
  --iteration 1 \
  --worker-id "$SLURM_ARRAY_TASK_ID" \
  --num-workers "$SLURM_ARRAY_TASK_COUNT"
