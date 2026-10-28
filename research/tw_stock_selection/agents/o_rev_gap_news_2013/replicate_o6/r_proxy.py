#!/usr/bin/env python3
"""獨立複製 o6：缺口格代理表與 G1 覆蓋（自寫，不 import 研究員 proxy.py）。
輸入：r_gap_cells.csv（r_inventory.py）、r_mentions.parquet（r_parse.py）、V4 凍結的 R_FIX／Y5_FIX、官方月營收水準、raw_quarterly。
輸出：r_proxy_gap_cells.csv，並與研究員 data/proxy_table_gap_cells.csv 比對。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TOP = HERE.parent; AG = TOP.parent
V4D = AG / 'o_v4_revenue_patch_official'
SWITCH = pd.Timestamp('2021-03-01')
SHIFT = '--shift' in sys.argv          # m−1 無值時退一個月（N3 一定退；此旗標讓 N1／N2 也退）

YL = pd.read_parquet(V4D / 'yoy_long_R_FIX.parquet'); YL.index = pd.PeriodIndex(YL.index, freq='M')
Y5 = pd.read_parquet(V4D / 'yoy_rev5y_FIX.parquet'); Y5.index = pd.PeriodIndex(Y5.index, freq='M')
# 官方水準（千元）
lv = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str})
lv['p'] = pd.PeriodIndex(lv.month.str[:7], freq='M')
lv = lv.sort_values('market', key=lambda s: s != 'sii').drop_duplicates(['code', 'p'])
LV = {(c, p): v for c, p, v in zip(lv.code, lv.p, lv.revenue_k_twd) if np.isfinite(v) and v > 0}
r5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype={'code': str})
for c, y, v in zip(r5.code, r5.ym, r5.revenue):
    p = pd.Period(y, 'M')
    if (c, p) not in LV and np.isfinite(v) and v > 0: LV[(c, p)] = v

M = pd.read_parquet(HERE / 'r_mentions.parquet')
M['p'] = pd.PeriodIndex(M.rev_ym, freq='M')
M = M.sort_values('ts')
MG = {k: g for k, g in M.groupby(['code', 'p'])}

def off_yoy(d, c, p):
    Y = Y5 if d >= SWITCH else YL
    if p in Y.index and c in Y.columns:
        v = Y.at[p, c]
        if np.isfinite(v): return float(v)
    return None

def news_val(d, c, p):
    cut = d + pd.Timedelta(hours=13, minutes=30)
    g = MG.get((c, p))
    if g is None: return None
    g = g[g.ts <= cut]
    if not len(g): return None
    gy = g[g.yoy.notna()]
    if len(gy): return float(gy.yoy.iloc[0])
    gl = g[g.level_k.notna()]
    if not len(gl): return None
    lvl = float(gl.level_k.iloc[0])
    b = LV.get((c, p - 12))
    if b is None:
        gb = MG.get((c, p - 12))
        if gb is not None:
            gb = gb[(gb.ts <= cut) & gb.level_k.notna()]
            if len(gb): b = float(gb.level_k.iloc[0])
    if b is None or not b > 0: return None
    return (lvl / b - 1) * 100

def month_vals(d, c, ms):
    out = []
    for p in ms:
        o = off_yoy(d, c, p)
        if o is not None: out.append((o, 'off')); continue
        n = news_val(d, c, p)
        out.append((n, 'news') if n is not None else (None, None))
    return out

# ---------------- 季報 Q ----------------
Q = pd.read_parquet(V4D / 'raw_quarterly.parquet'); Q = Q[~Q.suspect]
QV = {(c, q): v for c, q, v in zip(Q.code, Q.q, Q.rev_k)}
def q_deadline(q):
    y, k = int(q[:4]), int(q[-1])
    if k == 1: return pd.Timestamp(f'{y}-05-15 23:59')
    if k == 2: return pd.Timestamp(f'{y}-08-31 23:59') if y == 2013 else pd.Timestamp(f'{y}-08-14 23:59')
    if k == 3: return pd.Timestamp(f'{y}-11-14 23:59')
    return pd.Timestamp(f'{y + 1}-03-31 23:59')
QLIST = sorted(Q.q.unique(), key=lambda q: (int(q[:4]), int(q[-1])))
def q_months(q):
    y, k = int(q[:4]), int(q[-1])
    return [pd.Period(f'{y}-{3 * (k - 1) + i:02d}', 'M') for i in (1, 2, 3)]
def q_val(d, c):
    cut = d + pd.Timedelta(hours=13, minutes=30)
    # 法定期限 < 訊號日的最近一季（依日曆，不依資料是否存在）
    y = d.year + 1; k = 4
    while True:
        q = f'{y}Q{k}'
        if q_deadline(q) < cut: break
        k -= 1
        if k == 0: y -= 1; k = 4
    v = QV.get((c, q))
    if v is None: return None
    qb = f'{int(q[:4]) - 1}Q{q[-1]}'
    b = QV.get((c, qb))
    if b is None:
        ls = [LV.get((c, p)) for p in q_months(qb)]
        if all(x is not None for x in ls): b = sum(ls)
    if b is None or not b > 0: return None
    return (v / b - 1) * 100

def proxies(d, c):
    m = pd.Period(d, 'M')
    ms = [m - 1, m - 2, m - 3]
    vals = month_vals(d, c, ms)
    # N3：m−1 無值時比照 AB 退一個月
    v3 = vals
    if vals[0][0] is None:
        v3 = month_vals(d, c, [m - 2, m - 3, m - 4])
    out = {}
    out['N3'] = float(np.mean([v for v, _ in v3])) if all(v is not None for v, _ in v3) and any(k == 'news' for _, k in v3) else np.nan
    vv = v3 if SHIFT else vals
    have = [(v, k) for v, k in vv if v is not None]
    nn = any(k == 'news' for _, k in have)
    out['N2'] = float(np.mean([v for v, _ in have])) if len(have) >= 2 and nn else np.nan
    out['N1'] = float(np.mean([v for v, _ in have])) if len(have) >= 1 and nn else np.nan
    q = q_val(d, c)
    out['Q'] = q if q is not None else np.nan
    out['NQ'] = out['N1'] if np.isfinite(out['N1']) else out['Q']
    return out

if __name__ == '__main__':
    G = pd.read_csv(HERE / 'r_gap_cells.csv', dtype={'code': str})
    rows = []
    for r in G.itertuples():
        d = pd.Timestamp(r.sig)
        rows.append(dict(sig=r.sig, code=r.code, type=r.type, in_B10=r.in_B10, **{f'proxy_{k}': v for k, v in proxies(d, r.code).items()}))
    P = pd.DataFrame(rows)
    P.to_csv(HERE / ('r_proxy_gap_cells_shift.csv' if SHIFT else 'r_proxy_gap_cells.csv'), index=False)
    res = {}
    for k in ['N3', 'N2', 'N1', 'Q', 'NQ']:
        f = np.isfinite(P[f'proxy_{k}'])
        res[k] = dict(cov=float(f.mean()), cov_B10=float(f[P.in_B10].mean()))
    f = np.isfinite(P.proxy_N1)
    res['N1_by_type'] = P.assign(f=f).groupby('type').f.mean().round(4).to_dict()
    res['N1_by_year'] = P.assign(f=f, y=P.sig.str[:4]).groupby('y').f.mean().round(4).to_dict()
    res['N1_2013_2020'] = float(f[P.sig < '2021'].mean())
    res['N1_n'] = int(f.sum())
    print(json.dumps(res, ensure_ascii=False, indent=1))
    # 比對研究員
    R = pd.read_csv(TOP / 'data' / 'proxy_table_gap_cells.csv', dtype={'code': str})
    X = P.merge(R, on=['sig', 'code'], suffixes=('', '_r'))
    assert len(X) == len(P) == 732
    cmp = {}
    for k in ['N3', 'N2', 'N1', 'Q', 'NQ']:
        a, b = X[f'proxy_{k}'], X[f'proxy_{k}_r']
        fa, fb = np.isfinite(a), np.isfinite(b)
        both = fa & fb
        close = both & (np.abs(a - b) <= 0.01 + 1e-3 * np.abs(b))
        agree = (~fa & ~fb) | close
        cmp[k] = dict(cov_me=float(fa.mean()), cov_r=float(fb.mean()), diff_pp=float((fa.mean() - fb.mean()) * 100),
                      cell_agree=float(agree.mean()), cov_agree=float((fa == fb).mean()), n_both=int(both.sum()),
                      value_agree_when_both=float(close.sum() / max(1, both.sum())), only_me=int((fa & ~fb).sum()), only_r=int((~fa & fb).sum()))
    print(json.dumps(cmp, ensure_ascii=False, indent=1))
    X.to_csv(HERE / ('r_cmp_shift.csv' if SHIFT else 'r_cmp.csv'), index=False)
