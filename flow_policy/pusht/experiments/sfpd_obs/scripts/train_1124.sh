# Activate your Python virtual environment (if needed)
# source /home/sunsh16e/miniforge3/envs/robodiff/bin/python

# Navigate to the directory containing the script
cd /home/sunsh16e/flow-policy-1/flow_policy/pusht/experiments/sfpd_obs/

# Run the Python script with arguments
python train.py \
    --pred_horizon 17 \
    --seed 16 \
    --sin_embedding_scale 100 \
    --sigma 0.2 \
    --num_epochs 800 \
    --batch_size 256 \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --save_epoch 100 \
    --use_wandb \
    --model_save_dir "/home/sunsh16e/flow-policy-1/models"