#!/bin/sh
set -e
cd /home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1
python3 -I -B r1_main.py
python3 -I -B r2_tsmc.py
python3 -I -B r3_stats.py
python3 -I -B r4_pools.py
python3 -I -B r5_attr.py
python3 -I -B r5c_events.py
python3 -I -B r6_reg_cap.py
python3 -I -B r7_boot.py
python3 -I -B r8_compile.py
