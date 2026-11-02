#!/usr/bin/env python3
"""獨立複製 o8（o_rev_freshness_attention）：機制檢定的 α_g(m, j) 表。
只用共用引擎（v7env → V4 引擎、st_core 的 j 日曆與 AB），不讀研究員程式。
輸出 rp_alpha_s{seed}.parquet：每列 (leg, grp, m, j) 的淨／毛／壓力 α 與 n、N；rp_contrib_s{seed}.parquet：逐檔貢獻（A 指標）。"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
sys.path.insert(0, str(AG / 'o_v7_listing_exec_engine'))
import v7env
sys.path.pop(0)
L = v7env.load('v7'); K, V = L.K, L.V
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade')
sys.path.insert(0, str(AG / 'o_stagger_tranche'))
import st_core as STC
sys.path.pop(0)
IDX, COLS, N = K.IDX, K.COLS, K.N
seed = int(sys.argv[1]); VAR = sys.argv[2] if len(sys.argv) > 2 else ''
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)

OFF = L.V4(YFX, Y5F, DMT)
ANCH = [pd.Period(d, 'M') for d in OFF.sig_days]

class JV(L.V4):
    def y3(self, d, back=0):
        m = self.anchor[d] - back
        Y = self.Y5 if d >= V.SWITCH else self.YL
        y = K.yoy3_for_month(Y, m)
        return np.where(V.FINMASK, np.nan, y)

# ---- j 日曆 ----
DAYS = {}
for j in range(13):
    out = []
    for m in ANCH:
        d = STC.day_for(m, j)
        if d is None: continue
        t = IDX.get_loc(d)
        if t + 1 >= len(IDX) or not K.GOOD[t + 1]: continue
        out.append((m, d))
    de = STC.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or IDX.get_loc(de) + 1 >= len(IDX): te = len(IDX) - 1
    else: te = IDX.get_loc(de) + 1
    out = [(m, d) for m, d in out if IDX.get_loc(d) + 1 < te]
    DAYS[j] = (out, te)
anchor = {d: m for j in DAYS for (m, d) in DAYS[j][0]}
S = JV(YFX, Y5F, DMT, unavail=(lambda d: STC.unavail_for(anchor[d], d, seed)))
S.anchor = anchor
S52, V60 = S.S52, S.V60

# ---- 開盤對開盤報酬 ----
G = K.GROW
assert np.all(G > 0)
LG = np.vstack([np.zeros((1, N)), np.cumsum(np.log(G), axis=0)])   # LG[t+1] = sum_{<=t}
def oo_ret(e1, e2):
    """e1 開盤買、e2 開盤賣；停牌／下市後持平。"""
    tr1 = K.TRADED[e1] & np.isfinite(K.On[e1])
    a = np.where(tr1, K.Cn[e1] / np.where(tr1, K.On[e1], 1.0), 1.0)
    mid = np.exp(LG[e2 + 1] - LG[e1 + 1])        # prod G[e1+1..e2]
    tr2 = K.TRADED[e2] & np.isfinite(K.On[e2])
    b = np.where(tr2, K.On[e2] / np.where(tr2, K.Cn[e2], 1.0), 1.0)
    return a * mid * b - 1.0

# ---- 0050 PIT ----
TW = json.loads((AG / 'o_largecap_leg' / 'tw50_members.json').read_text(encoding='utf-8'))
PRE = json.loads((AG / 'o_capmatched_selection' / 'data' / 'membership_pre2015.json').read_text(encoding='utf-8'))
steps = sorted([(pd.Timestamp(s['eff']), set(s['members_before'])) for s in TW['steps_backward']])
cur = set(TW['current_2026-10-09'])
prek = sorted((pd.Timestamp(k), set(v)) for k, v in PRE.items())
def m50(x):
    if x >= pd.Timestamp('2015-09-18'):
        for eff, mb in steps:
            if eff > x: return mb
        return cur
    mem = None
    for k, v in prek:
        if k <= x: mem = v
    return mem
CI = K.CI
def m50mask(x):
    s = m50(x); m = np.zeros(N, bool)
    for c in s:
        if c in CI: m[CI[c]] = True
    return m
FO = pd.read_parquet(AG / 'o_sbl_foreign_level' / 'fo_panel.parquet').reindex(columns=COLS)
FOA = FO.to_numpy(dtype=float); FOI = FO.index
def fo_at(d):
    if d < pd.Timestamp('2021-10-08'): return None
    k = FOI.searchsorted(d, side='right') - 1
    return FOA[k]

BUY, SELL = K.rates(0.001)
SB, SS = 0.005, 0.005
LEGS = ('main', 'rev', 'px')
def order_by(score, y3, mask):
    s = np.round(np.where(mask, score, -9), 10)
    o = np.lexsort((np.arange(N), -np.where(mask & np.isfinite(y3), y3, -1e18), -s))
    return o[:int(mask.sum())]

rows = []; contrib = []
for j in range(13):
    days, te = DAYS[j]
    held = {}
    for i, (m, d) in enumerate(days):
        cs = S.cross_section(d)
        el = cs['el']; t, e = cs['t'], cs['e']
        e2 = IDX.get_loc(days[i + 1][1]) + 1 if i + 1 < len(days) else te
        r = oo_ret(e, e2)
        gu = K.gap_up_open(e)
        pr52 = pd.Series(np.where(el, cs['s52'], np.nan)).rank(pct=True).to_numpy()
        prr = pd.Series(np.where(el, cs['y3'], np.nan)).rank(pct=True).to_numpy()
        sc = {'main': 0.5 * pr52 + 0.5 * prr, 'rev': prr, 'px': pr52}
        v60 = V60[t]
        pv = pd.Series(np.where(el, v60, np.nan)).rank(pct=True).to_numpy()
        grp = {'A_lo': el & (pv <= 1/3), 'A_mid': el & (pv > 1/3) & (pv <= 2/3), 'A_hi': el & (pv > 2/3)}
        mm = m50mask(IDX[e] if VAR != 'sig' else d)
        grp['M50_in'] = el & mm; grp['M50_out'] = el & ~mm
        grp['COMBO'] = grp['A_lo'] & ~mm
        fo = fo_at(d) if VAR == '' else (fo_at(IDX[t - 1]) if VAR == 'sig' else fo_at(d))
        if VAR == 'sig' and d < pd.Timestamp('2021-10-08'): fo = None
        if fo is not None:
            elf = el & np.isfinite(fo)
            pf = pd.Series(np.where(elf, fo, np.nan)).rank(pct=True).to_numpy()
            grp['FO_lo'] = elf & (pf <= 1/3); grp['FO_mid'] = elf & (pf > 1/3) & (pf <= 2/3); grp['FO_hi'] = elf & (pf > 2/3)
        if VAR: grp = {k: v for k, v in grp.items() if k[:2] in ('M5', 'FO')}
        for g, gm in grp.items():
            Ng = int(gm.sum())
            if Ng == 0: continue
            ew = float(r[gm].mean())
            ng = max(1, int(round(0.2 * Ng)))
            for leg in LEGS:
                if leg != 'main' and not g.startswith('A_'): continue
                key = (g, leg)
                hp = held.get(key, set())
                cand = order_by(sc[leg], cs['y3'], gm)
                pick = []
                for q in cand:
                    q = int(q)
                    if gu[q] and q not in hp: continue
                    pick.append(q)
                    if len(pick) == ng: break
                pk = set(pick)
                pr = float(r[pick].mean())
                fb = len(pk - hp) / len(pk)
                fs = (len(hp - pk) / len(hp)) if hp else 0.0
                rows.append((leg, g, str(m), j, str(d.date()), pr - ew, pr - ew - BUY * fb - SELL * fs, pr - ew - SB * fb - SS * fs,
                             ng, Ng, fb, fs, pr, ew))
                held[key] = pk
                if leg == 'main' and g in ('A_lo', 'A_hi'):
                    for q in pick: contrib.append((g, str(m), j, COLS[q], float(r[q]) / ng))
                    for q in np.where(gm)[0]: contrib.append((g, str(m), j, COLS[q], -float(r[q]) / Ng))
    log('j', j, len(days))
df = pd.DataFrame(rows, columns=['leg', 'grp', 'm', 'j', 'd', 'a_gross', 'a_net', 'a_stress', 'n', 'N', 'fb', 'fs', 'pr', 'ew'])
df.to_parquet(HERE / f'rp_alpha_s{seed}{VAR}.parquet')
cdf = pd.DataFrame(contrib, columns=['grp', 'm', 'j', 'code', 'c'])
cdf = cdf.groupby(['grp', 'm', 'j', 'code'], as_index=False)['c'].sum()
cdf.to_parquet(HERE / f'rp_contrib_s{seed}{VAR}.parquet')
log('done', len(df))
