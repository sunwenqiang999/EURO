#!/bin/bash
# 查看训练任务的实时进度（支持 local_epochs 子目录）。
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=========== 总体状态 ==========="
tail -40 "$ROOT/sweep_main.log" 2>/dev/null || cat "$ROOT/train_logs_main.log" 2>/dev/null
if pgrep -f "run_all_training.sh|run_epochs_sweep.sh" > /dev/null; then
  echo "进程状态: 运行中"
else
  echo "进程状态: 已结束（或未启动）"
fi

echo ""
echo "=========== 各配置进度 ==========="
for sub in "" local_epochs1 local_epochs2; do
  logdir="$ROOT/train_logs${sub:+/$sub}"
  label="${sub:-local_epochs3(已完成)}"
  [ -d "$logdir" ] || continue
  echo "--- $label ---"
  for ds in mnist fmnist cifar10 cifar100; do
    f="$logdir/${ds}.log"
    if [ ! -f "$f" ]; then
      echo "    $ds: 未开始"
      continue
    fi
    last=$(grep "^Round" "$f" | tail -1)
    [ -z "$last" ] && echo "    $ds: 初始化中" || echo "    $ds: $last"
  done
done

echo ""
echo "=========== 已落盘产物 ==========="
for sub in local_epochs1 local_epochs2; do
  n_tl=$(ls -1 "$ROOT/timelist_new/$sub" 2>/dev/null | wc -l | tr -d ' ')
  n_md=$(ls -1 "$ROOT/models_new/$sub"/*.pth 2>/dev/null | wc -l | tr -d ' ')
  echo "  $sub: timelist $n_tl/8 个npy, 模型 $n_md/4 个pth"
done
