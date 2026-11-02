#!/bin/bash
# PREREG 後的補充執行：P3 去最大貢獻股、壓力與零成本診斷、G1 可執行版。
cd /home/user/-/research/tw_stock_selection/agents/o_capmatched_pit_close
until grep -q ALLDONE out/run_all.log; do sleep 20; done
python3 -I -B s3_run.py a MAIN --exclude 2615 >> out/run_more.log 2>&1
python3 -I -B s3_run.py a MAIN --cost stress >> out/run_more.log 2>&1
python3 -I -B s3_run.py a MAIN --cost zero >> out/run_more.log 2>&1
python3 -I -B s3_run.py b MAIN --cost zero >> out/run_more.log 2>&1
python3 -I -B s6_exec.py a >> out/run_more.log 2>&1
python3 -I -B s6_exec.py b >> out/run_more.log 2>&1
echo MOREDONE >> out/run_more.log
