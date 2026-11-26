#!/bin/bash
# 依次训练四个数据集，产出 timelist 曲线与模型。
#   mnist / fmnist  -> LeNet-5
#   cifar10/cifar100-> ResNet-18 (CIFAR 版)
#
# 环境变量：
#   LOCAL_EPOCHS  本地训练轮数（默认 3）。非 3 时产物写入 local_epochs<N>/ 子目录，
#                 避免覆盖已有的 local_epochs=3 结果。
#   ROUNDS        全局通信轮数（默认 50）
#   FORCE=1       即使模型已存在也强制重跑
#
# 用法：
#   ./run_all_training.sh                 # local_epochs=3，产物在 timelist_new/ 根下
#   LOCAL_EPOCHS=1 ./run_all_training.sh  # 产物在 timelist_new/local_epochs1/
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/.venv/bin/python"
ROUNDS="${ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"

# local_epochs=3 为历史默认，产物保留在根目录；其余取值各自建子目录区分。
if [ "$LOCAL_EPOCHS" = "3" ]; then
  SUB=""
else
  SUB="local_epochs${LOCAL_EPOCHS}"
fi
TL="$ROOT/timelist_new${SUB:+/$SUB}"
MD="$ROOT/models_new${SUB:+/$SUB}"
LOG="$ROOT/train_logs${SUB:+/$SUB}"

mkdir -p "$TL" "$MD" "$LOG"

run_one() {
  local ds="$1" model="$2"
  local ckpt="$MD/${ds}_${model}.pth"
  if [ -f "$ckpt" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$(date '+%H:%M:%S')] 跳过 $ds（已存在 $ckpt，设 FORCE=1 可强制重跑）"
    return 0
  fi
  echo "[$(date '+%H:%M:%S')] ===== 开始 $ds / $model / ${ROUNDS}轮 / 本地${LOCAL_EPOCHS}epoch ====="
  "$PY" "$ROOT/fl_training/train.py" \
      --dataset "$ds" --model "$model" --num-rounds "$ROUNDS" \
      --local-epochs "$LOCAL_EPOCHS" \
      --timelist-dir "$TL" --model-dir "$MD" \
      --run-name "${ds}_${model}_r${ROUNDS}_le${LOCAL_EPOCHS}" --no-figure \
      > "$LOG/${ds}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] ===== 完成 $ds（退出码 0）====="
  else
    echo "[$(date '+%H:%M:%S')] !!!!! $ds 失败，退出码 $rc，详见 $LOG/${ds}.log"
  fi
  return $rc
}

echo "[$(date '+%H:%M:%S')] 任务启动 | local_epochs=$LOCAL_EPOCHS | rounds=$ROUNDS"
echo "[$(date '+%H:%M:%S')] 产物目录: $TL 与 $MD"
run_one mnist    lenet5
run_one fmnist   lenet5
run_one cifar10  resnet18
run_one cifar100 resnet18
echo "[$(date '+%H:%M:%S')] 全部任务结束"
echo "--- $TL ---"; ls -la "$TL"
echo "--- $MD ---"; ls -la "$MD"
