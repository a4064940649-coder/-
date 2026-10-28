#!/usr/bin/env python3
"""複製員 o6：資料閘門、近似誤差（src 3 對官方）、開盤漲停遞補次數、可執行版、2006～2023 逐位相同。
用法：O6R_PANEL=compat python3 -I -B r11_misc.py   （O6R_PANEL=v4 只算遞補次數，用 V4 面板）"""
import os, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_env
PANEL = os.environ.get('O6R_PANEL', 'compat')
P4 = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
L = r4_env.load(P4 if PANEL == 'v4' else PANEL); K, V = L.K, L.V
OUT = HERE / f'out_{"compat" if PANEL == "v4" else PANEL}'
XR = L.setup_px()
YFX, Y5F = L.load_rfix(); GAP = L.gap_dict(); DMT = L.disp_mask_fn('trade')
def r(x, k=4): return None if x is None or not np.isfinite(x) else round(float(x), k)
res = {}

# ---------- 開盤漲停遞補（逐訊號日）----------
def blocked_by_date(S, n=10):
    out = {}; held = set()
    for d in S.sig_days:
        cs = S.cross_section(d); el = cs['el']
        if el.sum() < 50: continue
        gu = K.gap_up_open(cs['e']); cand = S.order(cs, 'official')[:int(el.sum())]
        pick = []; b = []
        for j in cand:
            if gu[j] and j not in held: b.append(K.COLS[j]); continue
            pick.append(j)
            if len(pick) == n: break
        if b: out[str(d.date())] = b
        held = set(pick)
    return out
for nm, kw in (('V', {}), ('V_B10', dict(gap=GAP, pfix=1.0)), ('AB_s0', dict(unavail=V.make_unavail('mc', 0)))):
    bb = blocked_by_date(L.V4(YFX, Y5F, DMT, **kw))
    res[f'blocked|{nm}'] = dict(total=sum(len(v) for v in bb.values()),
                                 win_2024_02_2025_09={d: v for d, v in bb.items() if '2024-02-01' <= d <= '2025-09-30'})
if PANEL == 'v4':
    json.dump(res, open(OUT / 'misc_v4.json', 'w'), ensure_ascii=False, indent=1); print(json.dumps(res, ensure_ascii=False)); sys.exit()

# ---------- 資料閘門：池內股日官方覆蓋 ----------
W = (K.IDX >= pd.Timestamp('2024-01-02')) & (K.IDX <= pd.Timestamp('2025-09-23'))
SRC = K.SRC.to_numpy()
S52, V60 = V.rolling_signals('gapfix')
def pool_at(t):
    v60 = V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    o = np.argsort(-np.where(base, v60, -np.inf), kind='stable')
    top = np.zeros(K.N, bool); top[o[:min(300, base.sum())]] = True
    return top & base
S = L.V4(YFX, Y5F, DMT)
sig = [d for d in S.sig_days + [S.end_day]]
def gate(kind):
    tot = 0; off = 0; nonoff = []
    for t in np.where(W)[0]:
        if kind == 'daily': pool = pool_at(t)
        else:
            dd = [d for d in sig if d <= K.IDX[t]]; pool = pool_at(K.IDX.get_loc(dd[-1]))
        m = pool & K.TRADED[t]
        tot += int(m.sum()); o = m & np.isin(SRC[t], [5, 6, 7]); off += int(o.sum())
        for j in np.where(m & ~o)[0]: nonoff.append((str(K.IDX[t].date()), K.COLS[j], float(SRC[t, j])))
    return tot, off, nonoff
for kind in ('monthly', 'daily'):
    tot, off, nonoff = gate(kind)
    res[f'gate_pool_{kind}'] = dict(stockdays=tot, official=off, pct=r(off / tot * 100, 4), nonofficial=nonoff[:10], n_nonofficial=len(nonoff))
# 2024-01-02～02-14
J = (K.IDX >= pd.Timestamp('2024-01-02')) & (K.IDX <= pd.Timestamp('2024-02-14'))
tot = off = 0
for t in np.where(J)[0]:
    dd = [d for d in sig if d <= K.IDX[t]]; pool = pool_at(K.IDX.get_loc(dd[-1])); m = pool & K.TRADED[t]
    tot += int(m.sum()); off += int((m & np.isin(SRC[t], [5, 6, 7])).sum())
res['gate_jan_feb'] = dict(stockdays=tot, pct=r(off / tot * 100))

# ---------- 所選格對其他官方來源的收盤一致率 ----------
D = HERE / 'data'
C = K.C
sel = pd.DataFrame(SRC, index=K.IDX, columns=K.COLS)
cells = []
for nm, code in (('hschi', 5), ('bacardi', 7), ('leochen', 7), ('tasi', 6)):
    s = pd.read_parquet(D / f'src_{nm}.parquet')
    if nm == 'bacardi': s = s[s['market'] == 'TWSE']
    s = s[(s['vol'] > 0) & s['code'].isin(set(K.COLS)) & (s['date'] >= '2024-01-02') & (s['date'] <= '2025-09-23')]
    s = s[['date', 'code', 'c']].rename(columns={'c': nm})
    cells.append(s.set_index(['date', 'code']))
A = pd.concat(cells, axis=1)
pc = C.stack(); pc.index.names = ['date', 'code']; pc = pc[(pc.index.get_level_values(0) >= '2024-01-02') & (pc.index.get_level_values(0) <= '2025-09-23')]
A = A.join(pc.rename('panel'), how='inner'); A = A[A['panel'].notna()]
ps = sel.stack(); ps.index.names = ['date', 'code']; A = A.join(ps.rename('src'), how='left')
# 每格：被選來源以外的其他官方來源
def others(row):
    pass
srcname = {5: ['hschi'], 7: ['bacardi', 'leochen'], 6: ['tasi']}
agree = tot = 0
ag_any = 0; tot_any = 0
for nm in ('hschi', 'bacardi', 'leochen', 'tasi'):
    x = A[A[nm].notna()]
    eq = (np.abs(x[nm] - x['panel']) < 1e-9)
    res[f'close_agree|{nm}'] = dict(n=int(len(x)), pct=r(eq.mean() * 100))
# 被選格：至少有另一來源者，與「全部其他來源」都一致的比例（排除自身來源）
own = {5: 'hschi', 6: 'tasi'}
def chosen(row_src, has):
    return None
A['own'] = np.where(A['src'] == 5, 'hschi', np.where(A['src'] == 6, 'tasi', np.where(A['src'] == 7, np.where(A['bacardi'].notna() & (np.abs(A['bacardi'] - A['panel']) < 1e-9), 'bacardi', 'leochen'), 'src3')))
oth = []
for nm in ('hschi', 'bacardi', 'leochen', 'tasi'):
    oth.append(np.where((A['own'] != nm) & A[nm].notna(), (np.abs(A[nm] - A['panel']) < 1e-9).astype(float), np.nan))
O = np.vstack(oth).T
has = np.isfinite(O).any(axis=1)
allok = np.nanmin(np.where(np.isfinite(O), O, 1.0), axis=1)
anyok = np.nanmax(np.where(np.isfinite(O), O, 0.0), axis=1)
res['close_agree_other_official'] = dict(n_with_other=int(has.sum()), all_agree_pct=r(allok[has].mean() * 100), any_agree_pct=r(anyok[has].mean() * 100),
                                         pairwise_pct=r(np.nanmean(O) * 100))
# ---------- 近似誤差：V4 src 3 對 V6 官方 ----------
P4C = pd.read_parquet(Path(P4) / 'close_raw.parquet').reindex(K.IDX); P4O = pd.read_parquet(Path(P4) / 'open.parquet').reindex(K.IDX)
P4S = pd.read_parquet(Path(P4) / 'src.parquet').reindex(K.IDX); P4R = pd.read_parquet(Path(P4) / 'ret.parquet').reindex(K.IDX)
m = (P4S.to_numpy() == 3) & np.isin(SRC, [5, 6, 7]) & W[:, None]
c4, c6 = P4C.to_numpy()[m], K.Cn[m]
o4, o6 = P4O.to_numpy()[m], K.On[m]
ok = np.isfinite(c4) & np.isfinite(c6)
res['approx_src3'] = dict(n=int(ok.sum()), close_absdiff_median_pct=r(np.median(np.abs(c4[ok] / c6[ok] - 1)) * 100),
                          close_identical_pct=r((np.abs(c4[ok] - c6[ok]) < 1e-9).mean() * 100),
                          open_identical_pct=r((np.abs(o4[np.isfinite(o4) & np.isfinite(o6)] - o6[np.isfinite(o4) & np.isfinite(o6)]) < 1e-9).mean() * 100))
r4 = P4R.to_numpy()[m]; r6 = K.Rn[m]; okr = np.isfinite(r4) & np.isfinite(r6)
res['approx_src3']['ret_corr'] = r(np.corrcoef(r4[okr], r6[okr])[0, 1])
# 前一筆也都是 src3 / 官方的純日報酬
# ---------- 2006～2023 逐位相同 ----------
NV = pd.read_parquet(OUT / 'navs.parquet'); V4N = pd.read_parquet(L.AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet')
for k6, k4 in (('V6|normal', 'V4|normal'), ('AB_V6|normal', 'AB_V4|normal'), ('AB_V6_B10|normal', 'AB_V4_B10|normal')):
    a = NV[k6].pct_change(); b = V4N[k4].pct_change()
    ix = a.index[(a.index <= pd.Timestamp('2023-12-29'))].intersection(b.index)
    d = (a.loc[ix] - b.loc[ix]).abs()
    first = (a.reindex(b.index) - b).abs()
    first = first[first > 0]
    res[f'identical_2006_2023|{k6}'] = dict(maxabs=float(d.max()), first_diff_date=str(first.index[0].date()) if len(first) else None)
# ---------- 可執行版 ----------
EX = pd.read_parquet(L.AG / 'o_exec_capacity_tiers' / 'navs_abavg.parquet')
BE = pd.read_parquet(L.AG / 'o_exec_capacity_tiers' / 'data' / 'bench.parquet')
sd = list(pd.DatetimeIndex(json.load(open(OUT / 'plans.json'))['sig_days']))
base = NV['AB_V6_B10|normal']; rb = base.pct_change()
ab4 = BE['AB_V4|normal'].pct_change()
exo = {}
for tier in ('30萬', '100萬', '300萬', '1000萬'):
    for fx in ('FIX', 'GROW'):
        drag = (EX[f'main|{tier}|{fx}|A'].pct_change() - ab4.reindex(EX.index)).reindex(base.index).fillna(0.0)
        nav = (1 + (rb + drag).fillna(0.0)).cumprod() * base.iloc[0]
        s = L.seg(nav, NV['AB_V6_B10|EW'], sd, '2006-04-01', '2026-10-12')
        exo[f'{tier}|{fx}'] = dict(CAGR=r(s['CAGR'], 2), t=r(s['t_0050'], 3))
        pd.DataFrame({'drag': drag}).to_parquet(OUT / f'execdrag_{tier}_{fx}.parquet')
res['exec_AB_V6_B10'] = exo
# 核對：o_exec 的 bench AB_V4 與 navs_v4 相同
res['exec_bench_check_maxabs'] = float((BE['AB_V4|normal'].reindex(V4N.index).pct_change() - V4N['AB_V4|normal'].pct_change()).abs().max())
json.dump(res, open(OUT / 'misc.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
