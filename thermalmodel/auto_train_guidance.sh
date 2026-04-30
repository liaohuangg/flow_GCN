#!/usr/bin/env bash
set -euo pipefail

# Auto training + ablation pipeline for ThermalGuidanceNet.
# Runs:
#   1) (optional) fp32 training
#   2) QAT training
#   3) Ablation eval:
#        - pick best fp32 ckpt (from checkpoints/fp32)
#        - pick best qat ckpt  (from checkpoints/qat)
#        - export PTQ-fp16 from best fp32
#        - report metrics + avg per-case GPU latency

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPOCHS=200
CKPT_EVERY=5
PRINT_EVERY=10

BS=8
LR="5e-4"
BASE=64
GRAD_W="0.1"
SEED=0

# fp16 export no longer uses calibration batches; keep flags for compatibility/no-op.
CALIB_BATCHES=50
CALIB_BS=8

AMP=0
AMP_DTYPE="fp16"

RUN_FP32=1
RUN_QAT=1
RUN_ABLATION=1

# Ablation eval / benchmark config (GPU by default)
EVAL_BS=8
BENCH_BS=8
WARMUP_BATCHES=10
TIMED_BATCHES=50
EVAL_DEVICE="cuda"

# Prefer running inside your conda env so dependencies match your manual runs.
# Default env name matches your earlier commands.
CONDA_ENV="chipdiffusion"

usage() {
  cat <<'EOF'
Usage: ./auto_train_guidance.sh [options]

Options:
  --epochs N           (default: 200)
  --ckpt-every N       (default: 5)
  --print-every N      (default: 10)
  --bs N               (default: 8)
  --lr LR              (default: 5e-4)
  --base N             (default: 64)
  --grad-w W           (default: 0.1)
  --seed N             (default: 0)
  --amp                enable mixed precision training (CUDA)
  --amp-dtype {fp16|bf16}  (default: fp16)
  --calib-batches N    (default: 50)
  --calib-bs N         (default: 8)
  --conda-env NAME     (default: chipdiffusion)

  --eval-bs N          eval batch size for metrics (default: 8)
  --bench-bs N         bench batch size for latency (default: 8)
  --warmup-batches N   warmup batches for bench (default: 10)
  --timed-batches N    timed batches for bench (default: 50)
  --eval-device {cuda|cpu}  (default: cuda)

  --no-fp32            skip fp32 training
  --no-qat             skip QAT training
  --no-ablation        skip ablation eval/benchmark

Examples:
  # Typical ablation run when fp32 is already trained:
  ./auto_train_guidance.sh --no-fp32

  # Change eval/bench knobs:
  ./auto_train_guidance.sh --no-fp32 --eval-bs 8 --bench-bs 8 --warmup-batches 10 --timed-batches 100
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2;;
    --ckpt-every) CKPT_EVERY="$2"; shift 2;;
    --print-every) PRINT_EVERY="$2"; shift 2;;
    --bs) BS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --base) BASE="$2"; shift 2;;
    --grad-w) GRAD_W="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --amp) AMP=1; shift 1;;
    --amp-dtype) AMP_DTYPE="$2"; shift 2;;
    --calib-batches) CALIB_BATCHES="$2"; shift 2;;
    --calib-bs) CALIB_BS="$2"; shift 2;;
    --conda-env) CONDA_ENV="$2"; shift 2;;

    --eval-bs) EVAL_BS="$2"; shift 2;;
    --bench-bs) BENCH_BS="$2"; shift 2;;
    --warmup-batches) WARMUP_BATCHES="$2"; shift 2;;
    --timed-batches) TIMED_BATCHES="$2"; shift 2;;
    --eval-device) EVAL_DEVICE="$2"; shift 2;;

    --no-fp32) RUN_FP32=0; shift 1;;
    --no-qat) RUN_QAT=0; shift 1;;
    --no-ablation) RUN_ABLATION=0; shift 1;;

    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

FP32_OUT="$DIR/checkpoints/fp32"
QAT_OUT="$DIR/checkpoints/qat"
PTQ_OUT="$DIR/checkpoints/ptq"
mkdir -p "$FP32_OUT" "$QAT_OUT" "$PTQ_OUT"

echo "[cfg] dir=$DIR"
echo "[cfg] epochs=$EPOCHS ckpt_every=$CKPT_EVERY print_every=$PRINT_EVERY"
echo "[cfg] bs=$BS lr=$LR base=$BASE grad_w=$GRAD_W seed=$SEED"
echo "[cfg] conda_env=$CONDA_ENV (uses: conda run --no-capture-output -n ... python -u)"
echo "[cfg] calib_batches=$CALIB_BATCHES calib_bs=$CALIB_BS (no-op for PTQ-fp16 export)"
if [[ "$AMP" -eq 1 ]]; then
  echo "[cfg] amp=1 amp_dtype=$AMP_DTYPE"
else
  echo "[cfg] amp=0"
fi

echo "[cfg] eval_device=$EVAL_DEVICE eval_bs=$EVAL_BS bench_bs=$BENCH_BS warmup_batches=$WARMUP_BATCHES timed_batches=$TIMED_BATCHES"

echo "[cfg] fp32_out=$FP32_OUT"
echo "[cfg] qat_out=$QAT_OUT"
echo "[cfg] ptq_out=$PTQ_OUT"

AMP_ARGS=()
if [[ "$AMP" -eq 1 ]]; then
  AMP_ARGS+=(--amp --amp_dtype "$AMP_DTYPE")
fi

RUNPY=(conda run --no-capture-output -n "$CONDA_ENV" python -u)

if [[ "$RUN_FP32" -eq 1 ]]; then
  echo "[run] fp32 training..."
  "${RUNPY[@]}" "$DIR/guidance_model.py" train \
    --epochs "$EPOCHS" \
    --batch_size "$BS" \
    --lr "$LR" \
    --base "$BASE" \
    --grad_w "$GRAD_W" \
    --seed "$SEED" \
    --ckpt_every "$CKPT_EVERY" \
    --print_every "$PRINT_EVERY" \
    --ckpt_tag fp32 \
    --out_dir "$FP32_OUT" \
    "${AMP_ARGS[@]}" \
    | tee "$LOG_DIR/train_fp32_${TS}.log"
else
  echo "[skip] fp32 training (using existing checkpoints in $FP32_OUT)"
fi

if [[ "$RUN_QAT" -eq 1 ]]; then
  echo "[run] QAT training..."
  "${RUNPY[@]}" "$DIR/guidance_model.py" train \
    --epochs "$EPOCHS" \
    --batch_size "$BS" \
    --lr "$LR" \
    --base "$BASE" \
    --grad_w "$GRAD_W" \
    --seed "$SEED" \
    --ckpt_every "$CKPT_EVERY" \
    --print_every "$PRINT_EVERY" \
    --qat \
    --ckpt_tag qat \
    --out_dir "$QAT_OUT" \
    "${AMP_ARGS[@]}" \
    | tee "$LOG_DIR/train_qat_${TS}.log"
else
  echo "[skip] QAT training"
fi

if [[ "$RUN_ABLATION" -eq 1 ]]; then
  echo "[run] ablation eval + PTQ-fp16 export + GPU benchmark..."
  OUT_JSON="$LOG_DIR/ablation_${TS}.json"
  "${RUNPY[@]}" "$DIR/ablation_eval.py" \
    --fp32_dir "$FP32_OUT" \
    --qat_dir "$QAT_OUT" \
    --ptq_dir "$PTQ_OUT" \
    --seed "$SEED" \
    --eval_bs "$EVAL_BS" \
    --bench_bs "$BENCH_BS" \
    --warmup_batches "$WARMUP_BATCHES" \
    --timed_batches "$TIMED_BATCHES" \
    --device "$EVAL_DEVICE" \
    --out_json "$OUT_JSON" \
    | tee "$LOG_DIR/ablation_${TS}.log"
  echo "[out] ablation_json=$OUT_JSON"
else
  echo "[skip] ablation eval"
fi

echo "[done]"

'''
训练时量化（QAT）：int8 QAT（quint8/qint8 observers），可选叠加 AMP(fp16/bf16)加速训练。
训练后量化（PTQ）：fp16 权重导出（model.half()），用于 GPU 推理加速。 
'''