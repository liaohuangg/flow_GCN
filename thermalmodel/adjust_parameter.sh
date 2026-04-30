#!/usr/bin/env bash
set -euo pipefail

# FP32 guidance model hyperparameter sweep.
# - Trains 5 configs (fp32 only)
# - Saves checkpoint every 5 epochs
# - Logs training stdout to logs/
# - After training, evaluates every saved checkpoint (every 5 epochs)
#   and writes metrics to eval_logs/ plus worst-case images to worst_figs/

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------- defaults (override via env vars) --------
CONDA_ENV="${CONDA_ENV:-chipdiffusion}"
EPOCHS="${EPOCHS:-200}"
BS="${BS:-8}"
SEED="${SEED:-0}"
CKPT_EVERY="${CKPT_EVERY:-5}"
PRINT_EVERY="${PRINT_EVERY:-10}"

EVAL_BS="${EVAL_BS:-8}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

LOG_DIR="$DIR/logs/fp32_sweep"
EVAL_LOG_DIR="$DIR/eval_logs/fp32_sweep"
CKPT_ROOT="$DIR/checkpoints/fp32_sweep"
FIG_ROOT="$DIR/worst_figs/fp32_sweep"

mkdir -p "$LOG_DIR" "$EVAL_LOG_DIR" "$CKPT_ROOT" "$FIG_ROOT"

RUNPY=(conda run --no-capture-output -n "$CONDA_ENV" python -u)

# -------- sweep configs --------
# format: "BASE LR GRAD_W"
CONFIGS=(
  "64 3e-4 0.1"
  "64 3e-4 0.02"
  "64 3e-4 0.0"
  "96 3e-4 0.02"
  "96 2e-4 0.02"
)

TS="$(date +%Y%m%d_%H%M%S)"

echo "[cfg] conda_env=$CONDA_ENV"
echo "[cfg] epochs=$EPOCHS bs=$BS seed=$SEED ckpt_every=$CKPT_EVERY print_every=$PRINT_EVERY"
echo "[cfg] eval_device=$EVAL_DEVICE eval_bs=$EVAL_BS"
echo "[cfg] ckpt_root=$CKPT_ROOT"
echo "[cfg] log_dir=$LOG_DIR eval_log_dir=$EVAL_LOG_DIR fig_root=$FIG_ROOT"

for cfg in "${CONFIGS[@]}"; do
  read -r BASE LR GRAD_W <<<"$cfg"

  # tokens for readable run IDs
  LR_TOK="${LR}"
  GW_TOK="${GRAD_W//./p}"

  RUN_ID="b${BASE}_lr${LR_TOK}_gw${GW_TOK}_seed${SEED}"

  OUT_DIR="$CKPT_ROOT/$RUN_ID"
  mkdir -p "$OUT_DIR"

  TRAIN_LOG="$LOG_DIR/train_${RUN_ID}_${TS}.log"
  EVAL_LOG="$EVAL_LOG_DIR/eval_${RUN_ID}_${TS}.log"

  CKPT_TAG="fp32_${RUN_ID}"

  echo "[run] train $RUN_ID"
  echo "[run] out_dir=$OUT_DIR"
  echo "[run] train_log=$TRAIN_LOG"
  echo "[run] eval_log=$EVAL_LOG"

  "${RUNPY[@]}" "$DIR/guidance_model.py" train \
    --epochs "$EPOCHS" \
    --batch_size "$BS" \
    --lr "$LR" \
    --base "$BASE" \
    --grad_w "$GRAD_W" \
    --seed "$SEED" \
    --ckpt_every "$CKPT_EVERY" \
    --print_every "$PRINT_EVERY" \
    --ckpt_tag "$CKPT_TAG" \
    --out_dir "$OUT_DIR" \
    | tee "$TRAIN_LOG"

  echo "[run] eval all checkpoints for $RUN_ID" | tee -a "$EVAL_LOG"

  # Evaluate every checkpoint saved (should be every 5 epochs)
  shopt -s nullglob
  CKPTS=("$OUT_DIR"/guidance_net_${CKPT_TAG}_ep*_seed*_bs*_lr*_base*_gw*.pth)
  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[warn] no checkpoints found in $OUT_DIR" | tee -a "$EVAL_LOG"
    continue
  fi

  IFS=$'\n' CKPTS_SORTED=($(printf "%s\n" "${CKPTS[@]}" | sort))
  unset IFS

  for ckpt in "${CKPTS_SORTED[@]}"; do
    ep="$(basename "$ckpt" | sed -n 's/.*_ep\([0-9][0-9][0-9][0-9]\)_.*/\1/p')"
    [[ -z "$ep" ]] && ep="unk"

    FIG_DIR="$FIG_ROOT/$RUN_ID/ep${ep}"
    mkdir -p "$FIG_DIR"

    echo "[eval] ep=$ep ckpt=$ckpt" | tee -a "$EVAL_LOG"
    "${RUNPY[@]}" "$DIR/eval_guidance_ckpt.py" \
      --ckpt "$ckpt" \
      --seed "$SEED" \
      --eval_bs "$EVAL_BS" \
      --device "$EVAL_DEVICE" \
      --out_fig_dir "$FIG_DIR" \
      | tee -a "$EVAL_LOG"
  done

done

echo "[done]"
