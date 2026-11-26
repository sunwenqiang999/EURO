#!/bin/bash
# 本地训练轮数（local_epochs）敏感性实验：依次跑 local_epochs = 1 和 2。
# 产物分别落在 timelist_new/local_epochs1/ 与 timelist_new/local_epochs2/，
# 与已有的 local_epochs=3 结果（timelist_new/ 根目录）互不覆盖。
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"

for le in 1 2; do
  echo ""
  echo "############################################################"
  echo "#  local_epochs = $le   ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "############################################################"
  LOCAL_EPOCHS=$le "$ROOT/run_all_training.sh"
done

echo ""
echo "[$(date '+%H:%M:%S')] ===== local_epochs 扫描全部结束 ====="
