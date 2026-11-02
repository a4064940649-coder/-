"""低劑量對照（臺灣50）：用 P1 凍結版 v4（PIT 股數＋修正 4763）在截止日預測的納入／刪除股，
三窗：W1 截止+1 開盤→公告前一日收盤；W2 →公告日收盤；W3 →生效前一日收盤。
對照：截止日同為流動性合格（60 日均值 ≥1,000 萬、價 ≥10）非成分、非事件股中，52 週比與 20 日報酬（標準化）距離最近的 3 檔，等權。
成本（K5T 近似）：每邊 max(0.1%＋0.5·σ60·√(Q/OA), 一跳/開盤)＋手續費 0.1425%×0.28；賣出稅 0.3%；壓力每邊再加 0.4%。Q＝100 萬元，OA＝5%×V60（開盤量代理）。
t 值按審核期分群（同一次審核的事件先平均）。這是描述性對照，不凍結、不做樣本外（2019 年後資料已被 etf_rebalance 看過）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import eflib as E
import s2_tw50_p1 as S2
ADJ, CL, OP, VAL = E.P('adj'), E.P('close_raw'), E.P('open'), E.P('value')
SRC = E.P('src')
good = CL.notna().sum(axis=1) >= 300
A_ = ADJ[good]; V_ = VAL[good]
r52 = (A_ / A_.rolling(250, min_periods=200).max()).reindex(ADJ.index).ffill()
r20 = (A_ / A_.shift(20) - 1).reindex(ADJ.index).ffill()
v60 = V_.rolling(60, min_periods=40).mean().reindex(ADJ.index).ffill()
sig = np.log(A_).diff().rolling(60, min_periods=40).std().reindex(ADJ.index).ffill()
def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def side_cost(c, t, Q=1e6, stress=False):
    o = OP.at[t, c]; s = sig.at[t, c] if np.isfinite(sig.at[t, c]) else 0.03
    oa = 0.05 * v60.at[t, c] if np.isfinite(v60.at[t, c]) else np.nan
    imp = 0.001 + (0.5 * s * np.sqrt(Q / oa) if np.isfinite(oa) and oa > 0 else 0)
    imp = max(imp, float(tick(o)) / o) if np.isfinite(o) and o > 0 else imp
    return imp + 0.001425 * 0.28 + (0.004 if stress else 0)
def wret(c, t0, t1):
    """t0 開盤買 → t1 收盤。"""
    try:
        o, c0, a0, a1 = OP.at[t0, c], CL.at[t0, c], ADJ.at[t0, c], ADJ.at[t1, c]
    except KeyError: return np.nan
    if not all(np.isfinite([o, c0, a0, a1])) or o <= 0: return np.nan
    return a1 / a0 * c0 / o - 1
def controls(C, excl, k=3):
    elig = [c for c in CL.columns if E.is_common(c) and c not in excl and np.isfinite(v60.at[C, c]) and v60.at[C, c] >= 1e7 and CL.at[C, c] >= 10
            and np.isfinite(r52.at[C, c]) and np.isfinite(r20.at[C, c])]
    X = pd.DataFrame({'a': r52.loc[C, elig], 'b': r20.loc[C, elig]})
    Z = (X - X.mean()) / X.std()
    return Z
def run(Q=1e6):
    R = E.reviews(); R = R[R.etf == 'TW50']
    ev = E.events(); ev = ev[ev.etf == 'TW50']
    rows = []
    for r in R.itertuples():
        if pd.Timestamp(r.eff) < pd.Timestamp('2015-09-18'): continue
        adds, dels, rk, mem = S2.predict(r.cutoff, patch=True)
        C = E.tday_le(r.cutoff); t0 = E.tday_after(C, 1)
        A = E.tday_le(r.ann); A1 = E.tday_le(pd.Timestamp(r.ann) - pd.Timedelta(days=1)); E1 = E.tday_le(pd.Timestamp(r.eff) - pd.Timedelta(days=1))
        evs = set(ev[ev.review == r.review].code) | set(adds) | set(dels)
        Z = controls(C, mem | evs)
        for side, lst in (('add', adds), ('del', dels)):
            for c in lst:
                if c not in CL.columns or not np.isfinite(r52.at[C, c]) or not np.isfinite(r20.at[C, c]): continue
                zc = ((r52.at[C, c] - Z.index.map(lambda x: r52.at[C, x]).to_numpy().mean()) if False else None)
                X = pd.DataFrame({'a': r52.loc[C, Z.index], 'b': r20.loc[C, Z.index]})
                mu, sd = X.mean(), X.std()
                z0 = np.array([(r52.at[C, c] - mu['a']) / sd['a'], (r20.at[C, c] - mu['b']) / sd['b']])
                dist = np.sqrt(((Z.to_numpy() - z0) ** 2).sum(axis=1))
                ctl = list(Z.index[np.argsort(dist)[:3]])
                w = rk[c] if c in rk.index else np.nan
                rec = dict(review=r.review, side=side, code=c, correct=bool(c in set(ev[(ev.review == r.review) & (ev.action == side)].code)),
                           mcap_rank=float(w), adv20=float(V_.loc[:C].tail(20)[c].mean()) if c in V_.columns else np.nan,
                           src3=bool((SRC.loc[t0:E1, c] == 3).any()) if c in SRC.columns else False, ctl=ctl)
                for wn, t1 in (('W1', A1), ('W2', A), ('W3', E1)):
                    if t1 <= t0: rec[wn] = np.nan; continue
                    g = wret(c, t0, t1); cr = np.nanmean([wret(x, t0, t1) for x in ctl])
                    cost = side_cost(c, t0, Q) + (side_cost(c, t1, Q) if c in OP.columns else 0.004) + 0.003
                    scost = side_cost(c, t0, Q, True) + side_cost(c, t1, Q, True) + 0.003
                    rec[wn + '_gross_ex'] = g - cr; rec[wn + '_net_ex'] = g - cr - cost; rec[wn + '_stress_ex'] = g - cr - scost
                rows.append(rec)
    return pd.DataFrame(rows)
def ctstat(df, col):
    g = df.groupby('review')[col].mean().dropna()
    n = len(g)
    if n < 2: return dict(n_ev=int(df[col].notna().sum()), n_rev=n, mean=float(g.mean()) if n else None, t=None)
    return dict(n_ev=int(df[col].notna().sum()), n_rev=n, mean=round(float(df[col].mean()) * 100, 3), median=round(float(df[col].median()) * 100, 3),
                rev_mean=round(float(g.mean()) * 100, 3), t=round(float(g.mean() / (g.std(ddof=1) / np.sqrt(n))), 2))
if __name__ == '__main__':
    df = run()
    df.to_csv(E.HERE / 'tw50_control_events.csv', index=False)
    res = {}
    for side in ('add', 'del'):
        d = df[df.side == side]
        # 劑量代理：權重 ∝ 市值名次對應的市值；相對劑量＝市值 ÷ ADV20（AUM 常數不影響審核內排序）
        for per, m in (('all', d.review >= '0'), ('sel_le2019-04', d.review <= '2019-03'), ('post2019', d.review >= '2019-06')):
            dd = d[m]
            for wn in ('W1', 'W2', 'W3'):
                for k in ('gross_ex', 'net_ex', 'stress_ex'):
                    res[f'{side}|{per}|{wn}|{k}'] = ctstat(dd, f'{wn}_{k}')
    for k, v in res.items():
        if '|W2|' in k or 'net_ex' in k: print(k, v)
    json.dump(res, open(E.HERE / 'tw50_control.json', 'w'), ensure_ascii=False, indent=1)
