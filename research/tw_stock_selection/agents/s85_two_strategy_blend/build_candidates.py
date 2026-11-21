#!/usr/bin/env python3
"""S85 第 0 步之二（不算任何合併結果）：把既有三個「同引擎、嚴成本」候選的原規格，改用正式管線的三個開關
（EXRIGHT_FIX／DEADLINE_FIX／INS15_MASK 全開，面板與 0050 序列同正式淨值）重跑標準路徑（100 萬、位移 0），讓資料修正口徑與正式淨值一致。
規則完全照原研究的程式庫（只 import，不改）：
  PEAD_Q＝S11 主格 Q（s11lib.make_signal_q kind='perp'）；DISP_D＝S14 主格 D（s14lib.make_signal_disp，第一次處置）；
  PULLBACK_P＝S15 主格 P（s15lib.make_signal_pb hold=20）。全部 W10_70（0050 底倉 30%＋衛星 70%、每檔＝衛星÷10、空位放 0050）。
OTC_SURGE 直接用 s68_otc_universe_oos/navs.csv.gz 的 OTC 欄（該題已是正式開關）。
輸出 cand_navs_official_switch.csv.gz（含各自的 stockfrac）與 cand_build_check.json（含與舊口徑淨值的差）。用法：python3 -I -B build_candidates.py"""
import sys, json, resource, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/offfix/cache')
t0 = time.time()
for d in ('s19_xq_threshold', 's15_strong_pullback', 's14_corporate_events', 's11_pead_second_signal'):
    sys.path.insert(0, str(AG / d))
import s19lib, s11lib, s14lib, s15lib
from s19lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS
def repoint():
    for mod, attr in ((s19lib, 'S19'), (s10lib, 'S10'), (s9lib, 'S9'), (s8lib, 'S8'), (s7lib, 'S7'), (s6lib, 'S6'), (s5lib, 'S5'),
                      (s4lib, 'S4'), (s3lib, 'S3'), (s2lib, 'S2'), (stlib, 'S1'), (cslib, 'TMP'), (s11lib, 'S11'), (s14lib, 'S14'), (s15lib, 'S15')):
        setattr(mod, attr, TMP)
repoint()
assert s11lib.s10lib is s10lib and s14lib.s10lib is s10lib and s15lib.s10lib is s10lib and s11lib.cslib is cslib
cslib.EXRIGHT_FIX = True; cslib.DEADLINE_FIX = True; cslib.INS15_MASK = True
P = s19lib.load(); s19lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
kw = dict(sat_w=0.7, sizing='W10', lock=False, cost=COSTS['strict'], capital=1e6)
out = {}; info = {}
# E0 驗收
S, _ = s19lib.rev_std(P); avA = s19lib.a10_rows(P, list(S.index))
nav, lg, st = s10lib.simulate_s10(P, s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=0), U, **kw)
off = pd.read_csv('/home/user/-/xq_scripts/site/navs_official.csv.gz', index_col=0, parse_dates=True)
info['E0_vs_official_max_abs'] = float(np.nanmax(np.abs(nav.reindex(off.index).to_numpy() - off['E0|W10_70'].to_numpy())))
assert info['E0_vs_official_max_abs'] < 0.01
def run(name, Sg):
    nav, lg, st = s10lib.simulate_s10(P, Sg, U, **kw)
    out[name] = nav; out[name + '|stockfrac'] = st['stockfrac']
    info[name] = dict(n_trades=int(len(lg)), final_nav=float(nav.iloc[-1]), mean_stockfrac=float(st['stockfrac'].mean()))
    print(name, info[name], f'{time.time()-t0:.0f}s', flush=True)
# DISP_D
DT = s14lib.disp_table(); CH = s14lib.disp_chains(P, DT, 'first')
ADV20 = pd.DataFrame(np.nan_to_num(P.AMT)).rolling(20, min_periods=20).mean().to_numpy()
run('DISP_D', s14lib.make_signal_disp(P, U, CH, shift=0, adv20=ADV20))
# PULLBACK_P
FF = s15lib.pb_features(P, U)
run('PULLBACK_P', s15lib.make_signal_pb(P, U, FF, shift=0, hold=20))
# PEAD_Q
FT = s11lib.fin_tables()
run('PEAD_Q', s11lib.make_signal_q(P, U, FT, shift=0, kind='perp'))
df = pd.DataFrame(out).loc[off.index[0]:off.index[-1]]; df.index.name = 'date'
df.to_csv(HERE / 'cand_navs_official_switch.csv.gz', float_format='%.6f')
# 與舊口徑（各題存檔）比較：只記錄期末淨值比
for name, f, c in (('DISP_D', 's14_corporate_events', 'D'), ('PULLBACK_P', 's15_strong_pullback', 'P'), ('PEAD_Q', 's11_pead_second_signal', 'Q')):
    old = pd.read_csv(AG / f / 'navs.csv.gz', index_col=0, parse_dates=True)[c]
    r = (df[name] / old.reindex(df.index)).dropna()
    info[name]['final_ratio_vs_old_file'] = float(r.iloc[-1]); info[name]['max_abs_ratio_dev_vs_old_file'] = float((r - 1).abs().max())
info['peak_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024); info['secs'] = round(time.time() - t0)
(HERE / 'cand_build_check.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(json.dumps(info, ensure_ascii=False))
