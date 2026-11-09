#!/usr/bin/env python3
"""o18 題 2 獨立複製（步驟 4）：重建 V7X AB_V7_B10 名單（13 j × 20 seeds），處置遮罩 orig（DMT）／union／replace。
只讀 import V7 既有接線：o_v7_listing_exec_engine/v7env.py（→ v4lib.V4）與 o_stagger_tranche/st_core（j 日期、AB unavail）。
JV7 子類別與 days_for 依 s4_jscan7.py 的定義改寫（只取 AB_V7_B10 名單、不模擬）。處置聯集／替換由複製員自寫。
輸出 scratchpad/o18_delistterms/replicate_o18/plans_{orig,union,replace}.csv；斷言 orig 與 plans_j7 逐列相同。"""
import sys, json, hashlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
O7 = AG / 'o_v7_listing_exec_engine'
WK = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o18_delistterms/replicate_o18'); WK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(O7))
import v7env
L = v7env.load('v7'); K, V = L.K, L.V
sys.path.pop(0)
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
sys.path.insert(0, str(AG / 'o_stagger_tranche'))
import st_core as STC
sys.path.pop(0)
OFF = L.V4(YFX, Y5F, DMT)
ANCH = [pd.Period(d, 'M') for d in OFF.sig_days]

DUF = AG / 'o_rev_avail_ds_pit_count' / 'raw' / 'disposal_official_universe.csv'
assert hashlib.sha256(DUF.read_bytes()).hexdigest().startswith('2842f45d')
DU = pd.read_csv(DUF, dtype={'code': str}, parse_dates=['start', 'end'])
CI = {c: j for j, c in enumerate(K.COLS)}
HI = pd.Timestamp('2021-09-23')
def off_cover(d):
    t = K.IDX.get_loc(d); e = K.IDX[t + 1] if t + 1 < len(K.IDX) else d
    x = DU[((DU.start <= d) & (DU.end >= d)) | ((DU.start <= e) & (DU.end >= e))]
    m = np.zeros(K.N, bool)
    for c in x.code:
        if c in CI: m[CI[c]] = True
    return m
_c = {}
def mk(kind):
    def f(d):
        d = pd.Timestamp(d); key = (kind, d)
        if key in _c: return _c[key]
        if kind == 'orig': m = DMT(d)
        elif kind == 'union': m = DMT(d) | off_cover(d) if pd.Timestamp('2006-01-01') <= d <= HI else DMT(d)
        elif kind == 'replace': m = off_cover(d) if pd.Timestamp('2014-01-01') <= d <= HI else DMT(d)
        _c[key] = m; return m
    return f
MASK = {k: mk(k) for k in ('orig', 'union', 'replace')}

class JV7(L.V4):
    def set_days(self, days, end_day):
        self.anchor = {d: m for m, d in days}; self.sig_days = [d for m, d in days]; self.end_day = end_day; self._cs = {}
    def y3(self, d, back=0):
        m = self.anchor[d] - back
        Y = self.Y5 if d >= V.SWITCH else self.YL
        y = K.yoy3_for_month(Y, m)
        if self.exfin: y = np.where(V.FINMASK, np.nan, y)
        return y

def days_for(j):
    out = []
    for m in ANCH:
        d = STC.day_for(m, j)
        if d is None: continue
        t = K.IDX.get_loc(d)
        if not K.GOOD[t + 1]: continue
        out.append((m, d))
    de = STC.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > K.IDX[-1]: de = K.IDX[-1]
    return [(m, d) for m, d in out if d < de], de

def job(a):
    kind, j, seed = a
    days, de = days_for(j)
    S = JV7(YFX, Y5F, MASK[kind], unavail=(lambda d, _s=seed: STC.unavail_for(S.anchor[d], d, _s)), gap=GAP, pfix=1.0)
    S.set_days(days, de)
    pl = S.plan(10, 'official')
    return kind, [('AB_V7_B10', j, seed, str(p[3].date()), str(K.IDX[p[0]].date()), int(p[0]), ','.join(K.COLS[jj] for jj in p[1])) for p in pl]

if __name__ == '__main__':
    kinds = sys.argv[1:] or ['orig', 'union', 'replace']
    jobs = [(k, j, s) for k in kinds for j in range(13) for s in range(20)]
    rows = {k: [] for k in kinds}
    with Pool(4) as P:
        for k, r in P.imap_unordered(job, jobs, chunksize=2): rows[k].extend(r)
    cols = ['strat', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks']
    for k in kinds:
        df = pd.DataFrame(rows[k], columns=cols).sort_values(['j', 'seed', 'exec_idx']).reset_index(drop=True)
        df.to_csv(WK / f'plans_{k}.csv', index=False); print(k, len(df), flush=True)
    if 'orig' in kinds:
        ref = pd.read_csv(O7 / 'data' / 'plans_j7.csv', dtype={'picks': str})
        ref = ref[(ref.strat == 'AB_V7_B10') & (ref.seed < 20)].sort_values(['j', 'seed', 'exec_idx']).reset_index(drop=True)
        mine = pd.read_csv(WK / 'plans_orig.csv', dtype={'picks': str})
        eq = (len(ref) == len(mine)) and bool((ref[cols].astype(str).to_numpy() == mine[cols].astype(str).to_numpy()).all())
        print('orig == plans_j7:', eq, len(ref), len(mine))
        (HERE / 'r4_check.json').write_text(json.dumps(dict(orig_equal_plans_j7=eq, n_ref=len(ref), n_mine=len(mine)), indent=1))
