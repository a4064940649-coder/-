#!/bin/bash
# S26 第三批：需要「別題結果」的腳本（S7 讀 S6、S8 讀 S6／S7、S24 讀 S19 與兩個資料夾），用 --link 指到修正後的重跑或原資料夾（唯讀）。
L=/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s26/logs
AG=/home/user/-/research/tw_stock_selection/agents
cd "$(dirname "$0")"
run() { tag=$1; shift; echo "=== $(date +%T) $tag"; timeout 5400 python3 -I -B runx.py "$@" > "$L/$tag.log" 2>&1; echo "rc=$? $(tail -1 $L/$tag.log)"; }
run s7_run_fix1 s7_locked_base run.py 1 s7_run_fix1 --link:s6_satellite_sizing=s6_run_fix1
run s8_run_fix1 s8_drawdown_cap run.py 1 s8_run_fix1 --link:s6_satellite_sizing=s6_run_fix1 --link:s7_locked_base=s7_run_fix1
run s24_build_fix1 s24_return_attribution build.py 1 s24_fix1 --path:s20_persistent_growth --link:s19_xq_threshold=s19_run_fix1 --link:o_lc_sector_rotation_pit=$AG/o_lc_sector_rotation_pit --link:o_pead_margin_surprise_low_attention=$AG/o_pead_margin_surprise_low_attention
run s24_analyze_fix1 s24_return_attribution analyze.py 1 s24_fix1 --keep
