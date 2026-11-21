#!/bin/bash
# 數據稽核部 2026-11-21：自己會開三個開關的題目（S46／S62／S64／S39）。用法：bash batch_self.sh <v1|v2> [題目...]
# v1＝完全照原研究（不預設任何開關）；v2＝只預先加開 REDUCE_FIX 與 0050 官方含息序列（題目自己再開另外三個＝五個全開；S39 只開除權息）
M=$1; shift
L=/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/v2recalc/logs; mkdir -p $L
AG=/home/user/-/research/tw_stock_selection/agents
cd "$(dirname "$0")"
run() { tag=$1; shift; echo "=== $(date +%T) $tag $*"; timeout 7200 python3 -I -B harness.py "$@" > "$L/$tag.log" 2>&1; echo "rc=$? $(tail -1 $L/$tag.log)"; }
for j in "$@"; do
  case $j in
    s46) run s46_$M s46_momentum_gate_family run.py $M s46_$M --self --link:deadline_engine_fix=$AG/deadline_engine_fix ;;
    s62) run s62_$M s62_small_capital_retail run.py $M s62_$M --self ;;
    s64) run s64_$M s64_retail_behavior_tax run.py $M s64_$M --self ;;
    s39) run s39_$M s39_first_print_revenue run.py $M s39_$M --self ;;
  esac
done
