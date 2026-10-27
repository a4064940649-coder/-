#!/usr/bin/env python3
"""複製員：事件層級 CAR（對同池等權、對 52 週比十分位配對），按公告月分群 t。"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
C = pd.read_parquet(P / 'close_raw.parquet'); R = pd.read_parquet(P / 'ret.parquet'); V = pd.read_parquet(P / 'value.parquet')
F = pd.read_parquet(P / 'flag.parquet')
IDX = C.index; COLS = list(C.columns); CI = {c: j for j, c in enumerate(COLS)}
Cn, Rn = C.to_numpy(), R.to_numpy()
TR = np.isfinite(Cn)
GROW = np.where(TR, 1 + np.nan_to_num(Rn), 1.0)
GOOD = TR.sum(1) >= 300
g = np.where(GOOD)[0]
A = np.where(TR, np.cumprod(GROW, 0), np.nan)
V60 = np.full(Cn.shape, np.nan); S52 = np.full(Cn.shape, np.nan)
V60[g] = pd.DataFrame(V.to_numpy()[g]).rolling(60, min_periods=50).mean().to_numpy()
hi = pd.DataFrame(A[g]).rolling(252, min_periods=200).max().to_numpy()
S52[g] = A[g] / hi
V60 = pd.DataFrame(V60).ffill().to_numpy(); S52 = pd.DataFrame(S52).ffill().to_numpy()
POOL = (V60 >= 1e7) & (Cn >= 10)
POOL = pd.DataFrame(POOL).shift(1).fillna(False).to_numpy().astype(bool)   # 前一日資格
# 每日報酬：極端值保護（|r|>60% 視為資料錯誤設 0）
Rc = np.where(np.isfinite(Rn) & (np.abs(Rn) < 0.6), Rn, np.nan)
EW = np.nanmean(np.where(POOL, Rc, np.nan), axis=1)
EW = np.nan_to_num(EW)
# 52 週比十分位（以前一日 S52 在池內分位）
S52p = pd.DataFrame(np.where(POOL, pd.DataFrame(S52).shift(1).to_numpy(), np.nan))
DEC = np.floor(S52p.rank(axis=1, pct=True).to_numpy() * 10 - 1e-9)
DR = np.zeros((len(IDX), 10))
for k in range(10):
    m = (DEC == k)
    with np.errstate(invalid='ignore'):
        DR[:, k] = np.nan_to_num(np.nanmean(np.where(m, Rc, np.nan), axis=1))
CUM_EW = np.concatenate([[0], np.cumsum(EW)]); CUM_DR = np.vstack([np.zeros(10), np.cumsum(DR, 0)])
RcZ = np.nan_to_num(Rc)
CUM_R = np.vstack([np.zeros(len(COLS)), np.cumsum(RcZ, 0)])

def day0(d):
    i = IDX.searchsorted(pd.Timestamp(d))
    return i if i < len(IDX) else None

def car(j, t0, a, b, bench='ew'):
    s, e = t0 + a, t0 + b
    if s < 1 or e >= len(IDX): return np.nan
    if not TR[t0, j] and not TR[min(t0 + 1, len(IDX) - 1), j]: return np.nan
    # 窗末之後要還有交易（非下市）
    if not TR[s:e + 1, j].any(): return np.nan
    r = CUM_R[e + 1, j] - CUM_R[s, j]
    if bench == 'ew':
        return r - (CUM_EW[e + 1] - CUM_EW[s])
    k = DEC[s - 1, j] if np.isfinite(DEC[s - 1, j]) else DEC[t0 - 1, j]
    if not np.isfinite(k): return np.nan
    k = int(k)
    return r - (CUM_DR[e + 1, k] - CUM_DR[s, k])

def ctstat(x, cl):
    x = np.asarray(x, float); m = np.isfinite(x); x = x[m]; cl = np.asarray(cl)[m]
    n = len(x)
    if n < 5: return dict(n=n)
    mu = x.mean(); u = x - mu
    G = pd.Series(u).groupby(cl).sum().to_numpy()
    ng = len(G)
    se = math.sqrt((G ** 2).sum() * ng / max(ng - 1, 1)) / n
    mm = pd.Series(x).groupby(cl).mean()
    tmm = mm.mean() / (mm.std(ddof=1) / math.sqrt(len(mm)))
    return dict(n=n, mean_pct=round(mu * 100, 2), t_cl=round(mu / se, 2), n_cl=ng, t_monthmean=round(float(tmm), 2), t_iid=round(mu / (x.std(ddof=1) / math.sqrt(n)), 2))

E = pd.read_csv(H / 'my_events.csv', dtype={'code': str}, parse_dates=['recv', 'eff', 'list_date', 'anchor'])
E = E[E['code'].isin(CI)]
# 除權日：生效後第一個 flag 1/2 日（≤120 天）
def exdate(r):
    if pd.isna(r.eff): return pd.NaT
    f = F[r.code]
    x = f[(f.index > r.eff) & (f.index <= r.eff + pd.Timedelta(days=120)) & f.isin([1, 2])]
    return x.index[0] if len(x) else pd.NaT
E['exd'] = pd.NaT
m = E.etype == 'SEO'
E.loc[m, 'exd'] = [exdate(r) for r in E[m].itertuples()]

PER = {'sel': ('2006-04-01', '2019-03-31'), 'sel05': ('2005-01-01', '2019-04-30'), 'oos': ('2019-05-01', '2026-10-12')}
TESTS = [('SEO_recv_0_2', 'SEO', 'recv', 0, 2), ('SEO_recv_1_60', 'SEO', 'recv', 1, 60),
         ('SEO_ex_m10_m1', 'SEO', 'exd', -10, -1), ('SEO_ex_0_20', 'SEO', 'exd', 0, 20),
         ('CB_list_m20_m6', 'CB', 'list_date', -20, -6), ('CB_list_m5_20', 'CB', 'list_date', -5, 20),
         ('CB_list_1_60', 'CB', 'list_date', 1, 60), ('CB_recv_1_60', 'CB', 'recv', 1, 60),
         ('SD_recv_1_60', 'SD', 'recv', 1, 60)]
out = {}
for per, (a, b) in PER.items():
    for name, et, col, x, y in TESTS:
        ee = E[(E.etype == et) & E[col].notna()]
        rows = []
        for r in ee.itertuples():
            d = getattr(r, col)
            if not (pd.Timestamp(a) <= d <= pd.Timestamp(b)): continue
            t0 = day0(d)
            if t0 is None: continue
            j = CI[r.code]
            inpool = bool(POOL[t0, j]) if True else True
            rows.append((car(j, t0, x, y, 'ew'), car(j, t0, x, y, 'dec'), str(IDX[t0].to_period('M')), inpool))
        if not rows: continue
        df = pd.DataFrame(rows, columns=['ew', 'dec', 'm', 'pool'])
        o = {}
        for lab, dd in (('all', df), ('pool', df[df.pool])):
            o[lab] = {'ew': ctstat(dd.ew, dd.m), 'dec': ctstat(dd.dec, dd.m)}
        out[f'{per}|{name}'] = o
        print(per, name, 'all', o['all']['ew'], '| dec', o['all']['dec'].get('mean_pct'), o['all']['dec'].get('t_cl'), '|| pool', o['pool']['ew'].get('n'), o['pool']['ew'].get('mean_pct'), o['pool']['ew'].get('t_cl'), '| dec', o['pool']['dec'].get('mean_pct'), o['pool']['dec'].get('t_cl'), flush=True)
# 挑選期可用事件數（收文日在挑選期、在面板內有交易）
avail = {}
for et in ('SEO', 'CB', 'SD'):
    ee = E[(E.etype == et)]
    d = ee['anchor']
    mm = (d >= '2006-04-01') & (d <= '2019-03-31')
    avail[et] = {'anchor_in_sel': int(mm.sum()), 'anchor_in_sel_pool': int(sum(POOL[day0(x), CI[c]] for x, c in zip(d[mm], ee['code'][mm]) if day0(x) is not None)),
                 'recv_2005_201904': int(((d >= '2005-01-01') & (d <= '2019-04-30')).sum())}
out['avail'] = avail
print(avail)
(H / 'my_evstudy.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
