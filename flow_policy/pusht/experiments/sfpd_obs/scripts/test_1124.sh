cd /home/sunsh16e/flow-policy-1/flow_policy/pusht/

# Set the default paths for the required parameters
MODEL_TYPE="sfpd"  # or sfps
CKPT_PATH="/home/sunsh16e/flow-policy/flow_policy/models/sigma_0.2_epoch_800_lr_4_optimizer_adam.pth"
#"/home/sunsh16e/flow-policy-1/models/sfpd_sigma_0.2_epoch_800_lr_4.pth"  
SAVE_DIR="/home/sunsh16e/flow-policy-1/flow_policy/pusht/experiments/sfpd_obs/results"  # Directory to save results
NAME="sfpd_sigma_0.2_epoch_800_lr_4"  # Name for this experiment

# Optional parameters with defaults
INTEGRATION_STEPS=1
NUM_TESTS=100
KP=500
KV=20
SEED=16
ENV_START_SEED=500
MAX_ROLLOUT_STEPS=200
OBS_HORIZON=2
ACTION_HORIZON=8
PRED_HORIZON=17
SIN_SCALE=100


# Run the Python script
python3 evaluate.py \
    --model-type "$MODEL_TYPE" \
    --ckpt-path "$CKPT_PATH" \
    --save-dir "$SAVE_DIR" \
    --name "$NAME" \
    --integration-steps-per-action "$INTEGRATION_STEPS" \
    --num-tests "$NUM_TESTS" \
    --kp "$KP" \
    --kv "$KV" \
    --seed "$SEED" \
    --env-start-seed "$ENV_START_SEED" \
    --max-rollout-steps "$MAX_ROLLOUT_STEPS" \
    --obs_horizon "$OBS_HORIZON" \
    --sin_embedding_scale "$SIN_SCALE" \