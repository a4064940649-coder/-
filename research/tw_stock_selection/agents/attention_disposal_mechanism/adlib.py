"""注意／處置機制研究共用函式（attention_disposal_mechanism）。

資料：
* 上市：面板 adj/close_raw/ret/value（2024-01-02～2024-02-14 缺口；跨缺口的事件一律剔除）。
* 上櫃補充：history.csv.gz（2025-09-24 起官方日 K），含息報酬用 exdiv_5y 的參考價修正（除權息日 ret = close/ref − 1）。
* 注意：attention_5y（date＝公告日，盤後已知）；處置：disposal_5y（date＝公告日，start/end＝處置期間）。
交易日曆 F＝面板日期 ∪ 注意公告日期 ∪ history 日期（2021-09 起），用來數「連續 N 個營業日」與「近 10 日 k 日」。
處置「鏈」：同一檔處置期間重疊或首尾相接（下一段 start ≤ 上一段 end 的下一個營業日）合併，鏈的最後一天＝真正解除處置前的最後交易日。
異常報酬＝個股含息報酬 − 同市場流動性合格（60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10，用前一日資訊）等權複利報酬。
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'); P = S / 'panel'
DD = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data'); X = DD / 'extras'
HERE = Path(__file__).resolve().parent
GAP = (pd.Timestamp('2024-01-02'), pd.Timestamp('2024-02-14'))
SPL = {'sel_2021-10_2023': ('2021-10-01', '2023-12-31'), 'val_2024_2026-10': ('2024-01-01', '2026-10-31'),
       'val_2024_2025': ('2024-01-01', '2025-12-31'), 'val_2026': ('2026-01-01', '2026-10-31')}
LIMIT_BUY, LIMIT_SELL = 0.09, -0.09


def load_events_raw():
    att = pd.read_csv(X / 'attention_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    dp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    return att, dp


def calendar(panel_idx, att, hist_dates):
    F = pd.DatetimeIndex(sorted(set(panel_idx[panel_idx >= '2021-06-01']) | set(att['date']) | set(hist_dates)))
    return F


def att_matrix(att, F, codes):
    M = np.zeros((len(F), len(codes)), dtype=bool)
    a = att[att['code'].isin(codes)]
    M[F.get_indexer(a['date']), pd.Index(codes).get_indexer(a['code'])] = True
    return pd.DataFrame(M, index=F, columns=codes)


def chains(dp, F):
    """回傳 list of dict(code, ann_i, s_i, e_i, chain_end_i) 與 coverage / ann 位置集合（F 上的索引）。"""
    out = []
    dp = dp.sort_values(['code', 'start', 'date'])
    for c, g in dp.groupby('code'):
        recs = []
        for _, r in g.iterrows():
            a_i = F.searchsorted(r['date'], 'right') - 1
            s_i = F.searchsorted(r['start']); e_i = F.searchsorted(r['end'], 'right') - 1
            if s_i >= len(F) or e_i < s_i: continue
            recs.append([a_i, s_i, e_i])
        recs.sort(key=lambda x: x[1])
        ch = []
        for a_i, s_i, e_i in recs:
            if ch and s_i <= ch[-1]['e'] + 1:
                ch[-1]['e'] = max(ch[-1]['e'], e_i); ch[-1]['recs'].append((a_i, s_i, e_i))
            else:
                ch.append({'s': s_i, 'e': e_i, 'recs': [(a_i, s_i, e_i)]})
        for k in ch:
            for a_i, s_i, e_i in k['recs']:
                out.append({'code': c, 'ann_i': a_i, 's_i': s_i, 'e_i': e_i, 'chain_s': k['s'], 'chain_e': k['e']})
    return pd.DataFrame(out)


def disposal_maps(DPc, F, codes):
    cov = np.zeros((len(F), len(codes)), dtype=bool); ann = np.zeros_like(cov)
    ann_chain_end = {}  # (code, ann_i) -> chain end i（只記「新鏈」的第一筆公告；鏈內追加公告也記）
    col = {c: j for j, c in enumerate(codes)}
    for r in DPc.itertuples():
        j = col.get(r.code)
        if j is None: continue
        cov[r.s_i:r.e_i + 1, j] = True; ann[r.ann_i, j] = True
        ann_chain_end[(r.code, r.ann_i)] = max(ann_chain_end.get((r.code, r.ann_i), -1), r.chain_e)
    return cov, ann, ann_chain_end


# ---------------- 價格資料 ----------------
def load_twse():
    D = {k: pd.read_parquet(P / f'{k}.parquet') for k in ['adj', 'close_raw', 'ret', 'value']}
    idx = D['adj'].index
    liq = ((D['value'].rolling(60, min_periods=60).mean() >= 1e7) & (D['close_raw'] >= 10)).shift(1, fill_value=False)
    R = D['ret']
    mkt = R.where(liq).mean(axis=1).fillna(0)
    return {'name': 'TWSE', 'idx': idx, 'adj': D['adj'], 'A': D['adj'].ffill(limit=60), 'ret': R, 'close': D['close_raw'],
            'liq': liq, 'mkt': mkt, 'LM': np.log1p(mkt.clip(lower=-0.99)).cumsum(), 'codes': list(D['adj'].columns)}


def load_tpex():
    h = pd.read_csv(DD / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    sl = pd.read_csv(DD / 'stock_list.csv', dtype={'code': str})
    tp = set(sl.loc[sl['market'] == 'TPEX', 'code'])
    h = h[h['code'].isin(tp) & (h['close'] > 0)]
    C = h.pivot(index='date', columns='code', values='close').sort_index()
    V = (h['close'] * h['volume']).groupby([h['date'], h['code']]).sum().unstack().reindex(C.index)
    ex = pd.read_csv(X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[(ex['market'] == 'TPEX') & ex['code'].isin(C.columns) & (ex['date'] >= C.index[0])]
    prev = C.shift(1)
    R = C / prev - 1
    for r in ex.itertuples():
        if r.date in C.index and r.ref_price > 0 and np.isfinite(C.at[r.date, r.code]):
            R.at[r.date, r.code] = C.at[r.date, r.code] / r.ref_price - 1
    R = R.where(R.abs() < 0.6)   # 極端值（公司行動、重新上市）視為缺值
    adj = (1 + R.fillna(0)).cumprod().where(C.notna())
    liq = ((V.rolling(60, min_periods=20).mean() >= 1e7) & (C >= 10)).shift(1, fill_value=False)
    mkt = R.where(liq).mean(axis=1).fillna(0)
    return {'name': 'TPEX', 'idx': C.index, 'adj': adj, 'A': adj.ffill(limit=60), 'ret': R, 'close': C, 'liq': liq,
            'mkt': mkt, 'LM': np.log1p(mkt.clip(lower=-0.99)).cumsum(), 'codes': list(C.columns)}


def spans_gap(d0, d1):
    return d0 <= GAP[1] and d1 >= GAP[0]


# ---------------- 交易模擬 ----------------
def simulate(M, plan, cov_F, F, codes_F, max_delay=20):
    """plan：list of dict(code, t (Timestamp 進場日), exit_F_i 或 hold_n, path)。
    進場 t 收盤；t 當日 ret ≥ 9% → 買不到。出場日 ret ≤ −9% → 順延到第一個非跌停日。
    成本：持有期間 (t, exit] 與任一處置期間重疊 → 0.8%，否則 0.5%。同一檔重疊事件只保留第一筆（買不到的不佔位）。"""
    idx = M['idx']; A = M['A'].to_numpy(); R = M['ret'].to_numpy(); LM = M['LM'].to_numpy(); LQ = M['liq'].to_numpy()
    col = {c: j for j, c in enumerate(M['codes'])}; colF = {c: j for j, c in enumerate(codes_F)}
    rows = []; busy = {}; drop = {'no_price': 0, 'gap': 0, 'overlap': 0, 'limit_up_entry': 0, 'no_exit_data': 0}
    for p in sorted(plan, key=lambda x: (x['t'], x['code'])):
        c = p['code']; j = col.get(c)
        if j is None or p['t'] not in idx: drop['no_price'] += 1; continue
        i0 = idx.get_loc(p['t'])
        if not np.isfinite(A[i0, j]):
            drop['no_price'] += 1; continue
        if p['t'] <= busy.get(c, pd.Timestamp('1900-01-01')): drop['overlap'] += 1; continue
        if 'exit_date' in p:
            if p['exit_date'] > idx[-1]: drop['no_exit_data'] += 1; continue
            if p['exit_date'] not in idx: drop['gap' if GAP[0] <= p['exit_date'] <= GAP[1] else 'no_exit_data'] += 1; continue
            i1 = idx.searchsorted(p['exit_date'], 'right') - 1
        else:
            i1 = i0 + p['hold_n']
        if i1 >= len(idx) or i1 <= i0: drop['no_exit_data'] += 1; continue
        if spans_gap(idx[i0], idx[i1]): drop['gap'] += 1; continue
        if np.isfinite(R[i0, j]) and R[i0, j] >= LIMIT_BUY: drop['limit_up_entry'] += 1; continue
        k = 0
        while i1 + 1 < len(idx) and k < max_delay and np.isfinite(R[i1, j]) and R[i1, j] <= LIMIT_SELL:
            i1 += 1; k += 1
        if spans_gap(idx[i0], idx[i1]): drop['gap'] += 1; continue
        if not np.isfinite(A[i1, j]): drop['no_exit_data'] += 1; continue
        raw = A[i1, j] / A[i0, j] - 1; m = math.expm1(LM[i1] - LM[i0])
        # 持有期間是否處於處置
        f0 = F.get_loc(idx[i0]) if idx[i0] in F else F.searchsorted(idx[i0]); f1 = F.searchsorted(idx[i1], 'right') - 1
        jf = colF.get(c)
        in_disp = bool(cov_F[f0 + 1:f1 + 1, jf].any()) if jf is not None and f1 > f0 else False
        cost = 0.008 if in_disp else 0.005
        busy[c] = idx[i1]
        rows.append({'date': idx[i0], 'code': c, 'exit': idx[i1], 'hold': i1 - i0, 'delay': k, 'path': p.get('path', ''),
                     'raw': raw, 'ab': raw - m, 'cost': cost, 'net': raw - m - cost, 'in_disp': in_disp,
                     'liq': bool(LQ[i0, j]), 'max_dd': float(np.nanmin(A[i0:i1 + 1, j] / A[i0, j] - 1))})
    return pd.DataFrame(rows), drop


# ---------------- 統計 ----------------
def tstat(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std() > 0 else None


def summ(df, col='net'):
    df = df.dropna(subset=[col])
    o = {'n': int(len(df))}
    if len(df) < 3: return o
    x = df[col]
    m = df.groupby(df['date'].dt.to_period('M'))[col].mean()
    lo, hi = x.quantile([0.05, 0.95])
    o.update({'mean_raw_pct': round(df['raw'].mean() * 100, 2) if 'raw' in df else None,
              'mean_ab_pct': round(df['ab'].mean() * 100, 2) if 'ab' in df else None,
              f'mean_{col}_pct': round(x.mean() * 100, 2), f'median_{col}_pct': round(x.median() * 100, 2),
              f'winsor5_mean_{col}_pct': round(x.clip(lo, hi).mean() * 100, 2),
              f'ex_top10_mean_{col}_pct': round(x.sort_values().iloc[:-10].mean() * 100, 2) if len(x) > 15 else None,
              'hit': round(float((x > 0).mean()), 3), 't_month': tstat(m), 't_date': tstat(df.groupby('date')[col].mean()),
              'n_months': int(len(m)), 'p5_pct': round(float(x.quantile(0.05)) * 100, 2), 'p10_pct': round(float(x.quantile(0.10)) * 100, 2),
              'min_pct': round(float(x.min()) * 100, 2), 'share_below_-10pct': round(float((x < -0.10).mean()), 3)})
    if 'in_disp' in df: o['share_in_disposal'] = round(float(df['in_disp'].mean()), 3)
    if 'hold' in df: o['mean_hold_days'] = round(float(df['hold'].mean()), 2)
    if 'max_dd' in df: o['mean_intratrade_min_pct'] = round(float(df['max_dd'].mean()) * 100, 2)
    return o


def by_period(df, col='net', splits=SPL):
    out = {'all': summ(df, col)}
    for k, (a, b) in splits.items():
        out[k] = summ(df[(df['date'] >= a) & (df['date'] <= b)], col)
    return out


def verdict(bp):
    s, v = bp.get('sel_2021-10_2023', {}), bp.get('val_2024_2026-10', {})
    sm, st = s.get('mean_net_pct'), s.get('t_month'); vm, vt = v.get('mean_net_pct'), v.get('t_month')
    if None in (sm, st, vm, vt): return 'insufficient'
    ok = sm > 0 and st >= 2 and vm > 0 and vt >= 2
    return 'PASS' if ok else 'FAIL'


def save(obj, name='results.json'):
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, pd.Timestamp): return str(o.date())
        raise TypeError(type(o))
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
