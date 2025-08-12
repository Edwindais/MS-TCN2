# #!/bin/bash

# python main.py --action=train --dataset=${1} --split=${2} \
#                 --num_epochs=100 \
                # --num_layers_PG=11 \
                # --num_layers_R=10 \
                # --num_R=3
# #!/bin/bash
DATASET="surgery_I3D_new"
MODEL_DIR="models_argumented"
RESULT_DIR="results_argumented"
# # DATASET="surgery_slowfast"
# # MODEL_DIR="models_slowfast"
# # RESULT_DIR="results_slowfast" 
EPOCH="200"
Dimension="1024"
SPLITS=(1 2 3 4 5)
# SPLITS=(1)

# echo "🚀 训练 Split 1 2 3 4 5..."
# for SPLIT in "${SPLITS[@]}"; do
#   echo "➡️ Training Split $SPLIT..."
#   CUDA_VISIBLE_DEVICES=1 python main.py \
#     --action train \
#     --dataset $DATASET \
#     --features_dim $Dimension \
#     --split $SPLIT \
#     --num_epochs $EPOCH \
#     --lr 3e-4 \
#     --use_wandb \
#     --wandb_project MSTCN++ \
#     --wandb_group MSTCN++ \
#     --num_layers_PG=13 \
#     --num_layers_R=12 \
#     --num_R=3
# done

# # ======== Step 2: Predict 每个 split =========
echo ""
echo "🔮 开始预测结果..."

CUDA_VISIBLE_DEVICES=1 python main.py \
  --action predict \
  --dataset $DATASET \
  --split 0 \
  --features_dim $Dimension \
  --num_epochs $EPOCH \
  --num_layers_PG=13 \
  --num_layers_R=12 \
  --num_R=3


# ======== Step 3  Eval 每个 split =========
# python eval.py --dataset $DATASET --split 0 --result_dir $RESULT_DIR




