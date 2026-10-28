#!/usr/bin/env python3
"""V5 clean-room 複製（研究員自做，不 import 研究員的 v5lib／v4lib／run_v5／jscan_v5）。
只讀：凍結資料（data_freeze_v5.json 所列檔案：data/fv_replace.csv 等）、frozen_spec 文字；
引擎＝o4 複製員的 replicate_o4/rcore.py（獨立實作：面板＋P、gapfix、處置成交日口徑、X、成本、二分法）；AB 照規格用 v2_core.make_unavail（mc, seed）。
比對：V5|normal、AB_V5|normal、逐 seed t、j＝0～12 的 t。"""
import sys, json, math, time
from pathlib import Path
from multiprocessing import Pool
AG = Path('/home/user/-/research/tw_stock_selection/agents')
HERE = AG / 'o_v5_pit_revenue'
sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official' / 'replicate_o4'))
import rcore as RC          # noqa: E402
sys.path.pop(0)
sys.path.insert(0, str(AG / 'main_v2'))
import v2_core as V2        # noqa: E402（規格指定的 AB 抽樣與延後機率表）
sys.path.pop(0)
import numpy as np, pandas as pd
assert list(V2.COLS) == list(RC.COLS)
IDX, COLS, N, CI = RC.IDX, RC.COLS, RC.N, RC.CI

# ---------- 首次值替換表 ----------
FR = pd.read_csv(HERE / 'data' / 'fv_replace.csv', dtype={'code': str})
NEWS = [(CI[c], pd.Period(y, 'M'), rl, r5, pd.Timestamp(fd)) for c, y, rl, r5, fd in zip(FR.code, FR.ym, FR.ratio_L, FR.ratio_5, FR.first_date) if c in CI]
def fv_y3(Y, m, d, panel, back=0):
    """V5 的 y3：m−1−back..m−3−back 三個月年增率，各格乘 f(本月)/f(去年同月)，f＝新聞首次值/最新值（文章日 < 訊號日才算）。"""
    ks = [m - 1 - back, m - 2 - back, m - 3 - back]
    if any(k not in Y.index for k in ks): return np.full(N, np.nan)
    Z = Y.loc[ks].to_numpy().copy()
    for j, ym, rl, r5, fd in NEWS:
        if not fd < d: continue
        r = rl if panel == 'L' else r5
        for i, k in enumerate(ks):
            if ym == k: Z[i, j] = ((1 + Z[i, j] / 100) * r - 1) * 100
            if ym == k - 12: Z[i, j] = ((1 + Z[i, j] / 100) / r - 1) * 100
    return np.where(np.isfinite(Z).all(0), Z.mean(0), np.nan)

# ---------- j 日曆（規格：錨定月 m 的期限＝m 月 10 日起第一個加權指數交易日；2019-02→2019-02-15、2023-04→2023-04-12 起第一個交易日）----------
TX = pd.DatetimeIndex(sorted(set(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date'])))
def first_td(ts):
    k = TX.searchsorted(ts); return TX[k] if k < len(TX) else None
def deadline(m):
    sp = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12')}
    return first_td(sp.get(str(m), pd.Timestamp(m.year, m.month, 10)))
def jday(m, j):
    D = deadline(m)
    if D is None: return None
    k = TX.get_loc(D) + j
    if k >= len(TX): return None
    s = TX[k]
    if s not in RC.TPOS or not RC.GOOD[RC.TPOS[s]]: return None
    return s
def unavail_j(m, s, seed):
    D = deadline(m); u = np.zeros(N, bool)
    if s < D: u[:] = True
    elif s == D or (str(m) == '2020-02' and s <= pd.Timestamp('2020-02-14')):
        u = np.random.default_rng([int(seed), int(s.strftime('%Y%m%d')), 0]).random(N) < V2.PCOL
    rm = str(m - 1)
    if '2026-01' <= rm <= '2026-08':
        D15 = first_td(pd.Timestamp(m.year, m.month, 15))
        if s < D15: u |= V2.IS15
        elif s == D15: u |= V2.IS15 & (np.random.default_rng([int(seed), int(s.strftime('%Y%m%d')), 15]).random(N) < V2.PCOL)
    return u

# ---------- 選股（rcore.build_plan 的同一規則，營收換成 fv_y3，訊號月可指定）----------
def plan(dates, unavail=None, bound=None):
    """dates＝[(i_d, i_e, m)]。bound：None／'B10'。"""
    out = []; held = set()
    for i_d, i_e, m in dates:
        d, e = IDX[i_d], IDX[i_e]
        Y, panel = (RC.YF, 'L') if d < RC.SWITCH else (RC.Y5, '5')
        yv = fv_y3(Y, m, d, panel)
        U = unavail(d, m) if unavail is not None else None
        if U is not None: yv = np.where(U, fv_y3(Y, m, d, panel, back=1), yv)
        v = RC.V60[i_d]; c = RC.Cn[i_d]
        base = np.isfinite(v) & (v >= 1e7) & np.isfinite(c) & (c >= 10)
        bi = np.where(base)[0]; bi = bi[np.argsort(-v[bi], kind='stable')][:300]
        top = np.zeros(N, bool); top[bi] = True
        ok = top & np.isfinite(RC.S52[i_d]) & np.isfinite(RC.On[i_e]) & np.isfinite(RC.Cn[i_e]) & ~RC.FINMASK & ~RC.FIN2MASK & ~RC.FORMASK
        ok &= ~RC.disposal_mask(d, e)
        gapadd = np.zeros(N, bool)
        if bound == 'B10':
            isgap = ~np.isnat(RC.GAPEND) & (RC.GAPEND >= np.datetime64(d)); gapadd = ok & ~np.isfinite(yv) & isgap
        pool = ok & (np.isfinite(yv) | gapadd); idx = np.where(pool)[0]
        if len(idx) < 50: continue
        s = pd.Series(RC.S52[i_d, idx]).rank(pct=True).to_numpy(); yy = yv[idx]; gg = gapadd[idx]
        if gg.any():
            py = np.full(len(idx), np.nan); py[~gg] = pd.Series(yy[~gg]).rank(pct=True).to_numpy(); py[gg] = 1.0
            yt = yy.copy(); yt[gg] = np.inf
        else:
            py = pd.Series(yy).rank(pct=True).to_numpy(); yt = yy
        sc = np.round(0.5 * s + 0.5 * py, 10)
        order = np.lexsort((idx, -np.where(np.isfinite(yt) | np.isinf(yt), yt, -np.inf), -sc))
        picks = []
        for k in order:
            j = idx[k]
            if j not in held and RC.limit_up(i_e, j): continue
            picks.append(j)
            if len(picks) == 10: break
        out.append(dict(i_d=i_d, i_e=i_e, picks=picks, pool=idx)); held = set(picks)
    return out

OFF = [(i_d, i_e, pd.Period(IDX[i_d], 'M')) for i_d, i_e in RC.signal_dates('F')]
B, S = RC.rates('normal')
def xret(nav, pl, end):
    ds = [p['i_d'] for p in pl]; de = ds[1:] + [RC.TPOS[end]]
    r = [(IDX[a], nav.iloc[b] / nav.iloc[a] - 1 - (RC.E50[b] / RC.E50[a] - 1)) for a, b in zip(ds, de) if pd.Timestamp('2006-04-01') <= IDX[a] <= RC.END]
    return pd.Series(dict(r))
def job_off(seed):
    U = None if seed is None else V2.make_unavail('mc', seed)
    pl = plan(OFF, unavail=(None if U is None else (lambda d, m, _U=U: _U(d))))
    nav, _ = RC.run(pl, B, S)
    return seed, nav, pl
def jdates(j):
    out = []
    for i_d, i_e, m in OFF:
        s = jday(m, j)
        if s is None: continue
        t = RC.TPOS[s]
        if not RC.GOOD[t + 1]: continue
        out.append((t, t + 1, m))
    de = jday(pd.Period('2026-10', 'M'), j)
    if de is None or de > IDX[-1]: de = IDX[-1]
    return [x for x in out if IDX[x[0]] < de], de
def job_j(a):
    j, seed = a
    dts, de = jdates(j)
    pl = plan(dts, unavail=lambda d, m, _s=seed: unavail_j(m, d, _s))
    nav, _ = RC.run(pl, B, S, end=de)
    return j, seed, nav, pl, de
if __name__ == '__main__':
    T0 = time.time()
    with Pool(4) as P: offs = P.map(job_off, [None] + list(range(40)), chunksize=1)
    print('off done', time.time() - T0, flush=True)
    with Pool(4) as P: js = P.map(job_j, [(j, s) for j in range(13) for s in range(40)], chunksize=2)
    print('j done', time.time() - T0, flush=True)
    NV = pd.read_parquet(HERE / 'navs_v5.parquet'); SN = pd.read_parquet(HERE / 'seed_navs_v5.parquet'); JN = pd.read_parquet(HERE / 'navs_jscan_v5.parquet')
    res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    a0 = pd.Timestamp('2006-04-11')
    def cmp(mine, rep):
        mine = mine.dropna(); rep = rep.reindex(mine.index); b0 = a0 if a0 in mine.index else mine.index[0]   # j≥2 的淨值起點晚於 2006-04-11
        return float((mine / mine[b0] / (rep / rep[b0]) - 1).abs().max())
    out = {}
    nav0 = offs[0][1]; out['V5|normal_maxreldiff'] = cmp(NV['V5|normal'], nav0)
    rets = [o[1].pct_change().fillna(0) for o in offs[1:]]
    ab = (1 + pd.concat(rets, axis=1).mean(1)).cumprod(); out['AB_V5|normal_maxreldiff'] = cmp(NV['AB_V5|normal'], ab)
    st = RC.summary(ab, offs[1][2]); out['AB_V5_rep'] = {'CAGR': st['cagr'], 't': st['t'], 'mdd': st['mdd']}
    seedt = [RC.tstat(xret(o[1], o[2], RC.END).values) for o in offs[1:]]
    out['seed_t_rep'] = seedt; out['seed_t_maxabsdiff'] = float(np.max(np.abs(np.array(seedt) - np.array(res['RESULTS']['seed_dist']['AB_V5|normal']['t_by_seed']))))
    out['seed_nav_maxreldiff'] = float(max(cmp(SN[f'AB_V5|normal|s{s}'], offs[1 + s][1]) for s in range(40)))
    jt = {}
    for j in range(13):
        lst = sorted([x for x in js if x[0] == j], key=lambda x: x[1]); de = lst[0][4]
        r = (1 + pd.concat([x[2].pct_change().fillna(0) for x in lst], axis=1).mean(1)).cumprod()
        jt[j] = {'t_rep': RC.tstat(xret(r, lst[0][3], de).values), 't_mine': res['RESULTS_JSCAN']['by_j'][str(j)]['t'], 'nav_maxreldiff': cmp(JN[f'J{j}'], r)}
    out['jscan'] = jt
    out['jscan_t_maxabsdiff'] = float(max(abs(v['t_rep'] - v['t_mine']) for v in jt.values()))
    out['pass_1e-9'] = bool(out['V5|normal_maxreldiff'] < 1e-9 and out['AB_V5|normal_maxreldiff'] < 1e-9 and out['seed_nav_maxreldiff'] < 1e-9 and max(v['nav_maxreldiff'] for v in jt.values()) < 1e-9)
    out['runtime_s'] = time.time() - T0
    json.dump(out, open(HERE / 'rep' / 'rep_v5.json', 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k not in ('seed_t_rep',)}, indent=1, default=str))
