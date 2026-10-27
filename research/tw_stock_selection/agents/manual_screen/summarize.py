#!/usr/bin/env python3
import json, sys
from pathlib import Path
r = json.loads((Path(__file__).resolve().parent / 'results_bt.json').read_text())
for tag in ('long', 'r5y'):
    for per, po in r[tag].items():
        if per in ('info', 'n_cand_median'): continue
        print(per)
        for k, o in po.items():
            print(f"  {k:10s} CAGR {o['CAGR']*100:6.1f} DD {o['maxDD']*100:6.1f} t50 {o['t_0050']:5.2f} tMain {o['t_vs_main'] if o['t_vs_main']==o['t_vs_main'] else 0:6.2f} tMx {o['t_vs_main_exfin'] if o['t_vs_main_exfin']==o['t_vs_main_exfin'] else 0:6.2f} tEW {o['t_EW']:5.2f} dM {o['diff_vs_main_m']*100:5.2f} ovl {o['overlap_main']:.2f}/{o['overlap_main_exfin']:.2f} held {o['avg_n_held']:.2f} lt10 {o['months_lt10']}/{o['n_sig']} str {o['CAGR_stress']*100:5.1f} 0050 {o['0050_CAGR']*100:5.1f}")
    print(tag, 'info', {k: v for k, v in r[tag]['info'].items()}, r[tag]['n_cand_median'])
