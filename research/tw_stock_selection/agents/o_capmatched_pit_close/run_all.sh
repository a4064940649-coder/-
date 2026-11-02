#!/bin/bash
# 依序執行（PREREG 後）。python3 -I -B。
cd /home/user/-/research/tw_stock_selection/agents/o_capmatched_pit_close
for c in i ii a b; do python3 -I -B s3_run.py $c MAIN >> out/run_all.log 2>&1; done
for c in a b o7 i ii; do python3 -I -B s3_run.py $c PRE >> out/run_all.log 2>&1; done
echo ALLDONE >> out/run_all.log
