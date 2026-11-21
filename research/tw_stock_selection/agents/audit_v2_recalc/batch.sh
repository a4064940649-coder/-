#!/bin/bash
# 數據稽核部 2026-11-21：依序跑各題。用法：bash batch.sh <v1|v2> [題目...]（v1＝改版前三開關；v2＝新口徑五開關）
M=$1; shift
L=/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/v2recalc/logs; mkdir -p $L
AG=/home/user/-/research/tw_stock_selection/agents
cd "$(dirname "$0")"
run() { tag=$1; shift; echo "=== $(date +%T) $tag $*"; timeout 7200 python3 -I -B harness.py "$@" > "$L/$tag.log" 2>&1; echo "rc=$? $(tail -1 $L/$tag.log)"; }
for j in "$@"; do
  case $j in
    s7)  run s7_$M s7_locked_base run.py $M s7_$M --link:s6_satellite_sizing=$AG/s6_satellite_sizing
         run s7p_$M s7_locked_base posthoc.py $M s7_$M --keep ;;
    s9)  run s9_$M s9_signal_timing run.py $M s9_$M ;;
    s27) run s27b_$M s27_expectation_ranges build.py $M s27_$M --link:exright_engine_fix=$AG/exright_engine_fix
         run s27a_$M s27_expectation_ranges analyze.py $M s27_$M --keep ;;
    s42) run s42_$M s42_realistic_expectation run.py $M s42_$M ;;
    s46) run s46_$M s46_momentum_gate_family run.py $M s46_$M --self
         run s46p_$M s46_momentum_gate_family posthoc.py $M s46_$M --keep --self ;;
    s62) run s62_$M s62_small_capital_retail run.py $M s62_$M --self
         run s62p_$M s62_small_capital_retail posthoc.py $M s62_$M --keep --self ;;
    s64) run s64_$M s64_retail_behavior_tax run.py $M s64_$M --self ;;
    s39) run s39_$M s39_first_print_revenue run.py $M s39_$M --self ;;
  esac
done
