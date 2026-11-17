#!/bin/bash
# 依序（單一行程）跑完：模式 exright（驗收用）與 official（正式）。用法：bash batch.sh <exright|official> [題目...]
# 題目：s7 s9 s27 s42 xas
M=$1; shift
L=/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/resync/logs; mkdir -p $L
AG=/home/user/-/research/tw_stock_selection/agents
cd "$(dirname "$0")"
run() { tag=$1; shift; echo "=== $(date +%T) $tag $*"; timeout 5400 python3 -I -B harness.py "$@" > "$L/$tag.log" 2>&1; echo "rc=$? $(tail -1 $L/$tag.log)"; }
for j in "$@"; do
  case $j in
    s7)  run s7_$M s7_locked_base run.py $M s7_$M --link:s6_satellite_sizing=$AG/s6_satellite_sizing
         run s7p_$M s7_locked_base posthoc.py $M s7_$M --keep ;;
    s9)  run s9_$M s9_signal_timing run.py $M s9_$M ;;
    s27) run s27b_$M s27_expectation_ranges build.py $M s27_$M --link:exright_engine_fix=$AG/exright_engine_fix
         run s27a_$M s27_expectation_ranges analyze.py $M s27_$M --keep ;;
    s42) run s42_$M s42_realistic_expectation run.py $M s42_$M ;;
    xas) run xas_$M exright_engine_fix run_engine.py $M xas_$M -- 1 XAS ;;
  esac
done
