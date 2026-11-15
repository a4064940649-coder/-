# -*- coding: utf-8 -*-
"""S19 獨立複製：凍結後診斷（分歧來源）。
腳本股池加上研究版股池的「近 60 列都有資料（停牌排除，rep_s2 full60）」條件——研究員的 XT 實際有這個條件，20 號腳本沒有——
其他完全照 rep_s19_run.py，整套重跑，輸出到 <out_dir>（不覆寫本資料夾的 my_numbers.json）。
用法：python3 -I -B rep_s19_f60.py <scratch_dir> <out_dir>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rep_s19_run as RR
K = RR.K
_orig = K.script_universe
K.script_universe = lambda P, s, ex: _orig(P, s, ex) & P.full60[s]
RR.main(sys.argv[1], odir=sys.argv[2])
