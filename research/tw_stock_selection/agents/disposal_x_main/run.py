#!/usr/bin/env python3
"""處置股 × 主策略排名：2006-04～2021-09 近似處置事件的樣本外檢驗。規則見 results.json PREREG（prereg.py 先寫入）。

事件建構與執行濾網逐字複製自 agents/disposal_bear_longhistory/run.py（該檔 import 時會跑完整組合模擬，故不直接 import）。
分組用 revenue_longhistory/lhlib.py 的 s52_fix／v60_fix／yoy_long；組合用 exec_rebuild/dlib.Sleeve。
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'agents' / 'exec_rebuild'))
sys.path.insert(0, str(ROOT / 'agents' / 'revenue_longhistory'))
import exlib as L, dlib as Dl                                    # noqa: E402
import lhlib as LH                                               # noqa: E402

RES = HERE / 'results.json'
res = json.loads(RES.read_text(encoding='utf-8'))
assert 'PREREG' in res and res['PREREG']['寫於執行前']

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
adj, C, V, R1 = L.adj, L.C, L.V, L.RET
assert adj.index.equals(LH.IDX) and adj.columns.equals(LH.COLS)
FLAG = pd.read_parquet(P / 'flag.parquet').reindex(index=adj.index, columns=adj.columns).to_numpy()
idx = adj.index; cols = adj.columns; CI = L.CI; T = len(idx)
RETN, ADJN = R1.to_numpy(), adj.to_numpy()
LIM = np.where(idx < pd.Timestamp('2015-06-01'), 0.07, 0.10)

# ---- 加權指數、200 日均線（同原檔）----
twr = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
ma = twr.rolling(200).mean()
bear_tw = (twr < ma).where(ma.notna())
tx = twr.reindex(idx).ffill()
bear_day = bear_tw.reindex(idx, method='ffill')
BEAR_ENTRY = np.r_[np.nan, bear_day.to_numpy(dtype=float)[:-1]]

# ---- 對照池（同原檔）----
liq = (V.rolling(60, min_periods=60).mean() >= 1e7) & (C >= 10)
mkt = R1.where(liq).mean(axis=1).fillna(0)
LA = np.log1p(R1.fillna(0).clip(lower=-0.99)).cumsum().to_numpy()
LM = np.log1p(mkt.clip(lower=-0.99)).cumsum().to_numpy()
LIQN = liq.to_numpy()
lastvalid = np.array([np.max(np.where(np.isfinite(ADJN[:, j]))[0]) if np.isfinite(ADJN[:, j]).any() else -1 for j in range(len(cols))])

# ---- 近似事件（同原檔）----
A6 = adj / adj.shift(6) - 1
m6 = tx / tx.shift(6) - 1
att = (A6 >= 0.32) & (A6.sub(m6, axis=0) >= 0.20)
trig = att & att.shift(1, fill_value=False) & att.shift(2, fill_value=False)
TR = trig.to_numpy()
prox = []
for c in range(len(cols)):
    i = 0; col = TR[:, c]
    while i < T:
        if col[i]:
            s, e = i + 1, i + 10
            if e < T: prox.append((cols[c], s, e))
            i = e + 1
        else:
            i += 1
prox = pd.DataFrame(prox, columns=['code', 'i0', 'i1']); prox['start'] = idx[prox.i0]; prox['src'] = 'proxy'

def execute(ev, entry_filter=True):
    ev = ev.sort_values(['i0']).reset_index(drop=True)
    kept_exit = {}; rows = []; st = {'raw': len(ev), 'no_price_entry': 0, 'untradable_entry': 0, 'overlap': 0, 'delayed_exit': 0, 'delisted_in_hold': 0}
    for r in ev.itertuples():
        c = r.code; j = CI[c]; i0, i1 = r.i0, r.i1
        if not np.isfinite(ADJN[i0, j]): st['no_price_entry'] += 1; continue
        if c in kept_exit and i0 <= kept_exit[c]: st['overlap'] += 1; continue
        if entry_filter and RETN[i0, j] >= LIM[i0] - 0.01: st['untradable_entry'] += 1; continue
        k = i1; dl = False
        if lastvalid[j] < i1:
            dl = True; st['delisted_in_hold'] += 1
        else:
            while k < lastvalid[j] and not (np.isfinite(RETN[k, j]) and RETN[k, j] > -(LIM[k] - 0.006)): k += 1
            if k > i1: st['delayed_exit'] += 1
        kept_exit[c] = k
        raw = math.expm1(LA[k, j] - LA[i0, j]); m = math.expm1(LM[k] - LM[i0])
        fl = FLAG[i0 + 1:k + 1, j]; fl = fl[np.isfinite(fl)]
        rows.append({'code': c, 'j': j, 'i0': i0, 'i1': i1, 'k': k, 'date': idx[i0], 'src': r.src, 'raw': raw, 'ab': raw - m,
                     'bear': BEAR_ENTRY[i0], 'liq': bool(LIQN[i0 - 1, j]) if i0 > 0 else False, 'delisted_in_hold': dl,
                     'corp_flag': bool(np.isin(fl, [3, 4, 5]).any()), 'v_in': L.VN[i0, j], 'v_out': L.VN[k, j]})
    return pd.DataFrame(rows), st

EP, stP = execute(prox)

# ---- 分組：資訊日＝進場日前一交易日 ----
YOY = LH.yoy_long(); Y3 = YOY.rolling(3, min_periods=3).mean().reindex(columns=cols)
GOODN = LH.GOOD.to_numpy()
S52 = LH.s52_fix.to_numpy(); V60 = LH.v60_fix.to_numpy(); CN = C.to_numpy()
cover = {'yoy_long_first_month': str(YOY.index.min()), 'yoy3_first_valid_month': str(Y3.dropna(how='all').index.min()),
         'yoy_long_codes_by_year': {int(y): int(Y3[Y3.index.year == y].notna().any().sum()) for y in range(2006, 2027)}}

def classify(row):
    i = row.i0 - 1
    while i > 0 and not GOODN[i]: i -= 1
    d0 = idx[row.i0]
    m = pd.Period(d0, 'M') - (2 if d0.day <= 10 else 1)
    v60 = V60[i]; px = CN[i]
    base = (v60 >= 1e7) & (px >= 10)
    vb = np.where(base, v60, np.nan)
    rk300 = pd.Series(vb).rank(ascending=False).to_numpy()
    t300 = base & (rk300 <= 300)
    y = Y3.loc[m].to_numpy() if m in Y3.index else np.full(len(cols), np.nan)
    s52 = S52[i]
    el = t300 & np.isfinite(y) & np.isfinite(s52)
    j = row.j
    info = {'info_date': idx[i], 'rev_month': str(m), 'in_top300': bool(t300[j]), 'has_rev': bool(np.isfinite(y[j])), 's52': s52[j], 'yoy3': y[j], 'pool_n': int(el.sum())}
    if not el[j]:
        info.update(grp='G3', main_rank=np.nan); return info
    sc = 0.5 * pd.Series(np.where(el, s52, np.nan)).rank(pct=True) + 0.5 * pd.Series(np.where(el, y, np.nan)).rank(pct=True)
    rnk = float(sc.rank(ascending=False).iloc[j])
    info.update(grp='G1' if rnk <= 100 else 'G2', main_rank=rnk); return info

EV = EP[(EP.date >= '2006-04-01') & (EP.date <= '2026-09-30')].copy().reset_index(drop=True)
info = pd.DataFrame([classify(r) for r in EV.itertuples()])
EV = pd.concat([EV, info], axis=1)
EV['net'] = EV['ab'] - 0.008; EV['net12'] = EV['ab'] - 0.012; EV['m'] = EV['date'].dt.to_period('M')
MAIN = EV[EV.date <= '2021-09-30']
BRIDGE = EV[EV.date >= '2021-10-01']

def tstat(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std() > 0 else None

def st(e, col='net'):
    if len(e) == 0: return {'n': 0}
    x = e[col]; mm = x.groupby(e['m']).mean()
    o = {'n': int(len(e)), 'mean_net_pct': round(float(x.mean() * 100), 2), 'median_net_pct': round(float(x.median() * 100), 2),
         'win': round(float((x > 0).mean()), 3), 't_month': tstat(mm), 'months': int(len(mm)),
         'mean_net_ex_top5pct': round(float(x[x <= x.quantile(0.95)].mean() * 100), 2) if len(e) >= 5 else None,
         'mean_raw_pct': round(float(e['raw'].mean() * 100), 2)}
    if col == 'net': o['mean_net_1.2_pct'] = round(float(e['net12'].mean() * 100), 2)
    return o

def diff(e, a='G1', col='net'):
    A = e[e.grp == a]; B = e[e.grp != a]
    if len(A) < 3 or len(B) < 3: return {}
    se = math.sqrt(A[col].var(ddof=1) / len(A) + B[col].var(ddof=1) / len(B))
    ma_, mb_ = A.groupby('m')[col].mean(), B.groupby('m')[col].mean(); dd = (ma_ - mb_).dropna()
    A2 = e[e.grp == 'G2']
    o = {'diff_G1_minus_rest_pct': round(float((A[col].mean() - B[col].mean()) * 100), 2), 'welch_t': round(float((A[col].mean() - B[col].mean()) / se), 2),
         'month_clustered_diff_t': tstat(dd), 'month_clustered_diff_mean_pct': round(float(dd.mean() * 100), 2) if len(dd) else None, 'months_both': int(len(dd))}
    if len(A2) >= 3:
        dd2 = (ma_ - A2.groupby('m')[col].mean()).dropna()
        o['G1_minus_G2_pct'] = round(float((A[col].mean() - A2[col].mean()) * 100), 2); o['G1_minus_G2_month_t'] = tstat(dd2); o['G1_minus_G2_months'] = int(len(dd2))
    return o

def block(e):
    o = {g: st(e[e.grp == g]) for g in ('G1', 'G2', 'G3')}
    o['G2uG3'] = st(e[e.grp != 'G1']); o['all'] = st(e); o['diff'] = diff(e)
    return o

SEG = {'MAIN_2006-04_2021-09': MAIN,
       'clean_2006-04_2012-12': MAIN[MAIN.date <= '2012-12-31'],
       'surv_2013-01_2021-09': MAIN[MAIN.date >= '2013-01-01'],
       'exIFRS_2014-01_2021-09': MAIN[MAIN.date >= '2014-01-01'],
       '7pct_2006-04_2015-05': MAIN[MAIN.date <= '2015-05-31'],
       '10pct_2015-06_2021-09': MAIN[MAIN.date >= '2015-06-01'],
       'bear_MA200': MAIN[MAIN.bear == 1], 'bull_MA200': MAIN[MAIN.bear == 0],
       'liquid_at_entry': MAIN[MAIN.liq], 'no_corp_flag': MAIN[~MAIN.corp_flag],
       'with_delisted_2006-04_2019-05-16': MAIN[MAIN.date <= '2019-05-16'],
       'BRIDGE_proxy_2021-10_2026-09': BRIDGE,
       'BRIDGE_proxy_2021-10_2023-12': BRIDGE[BRIDGE.date <= '2023-12-31'],
       'BRIDGE_proxy_2024-01_2026-09': BRIDGE[BRIDGE.date >= '2024-01-01']}
out = {k: block(v) for k, v in SEG.items()}

g1 = out['MAIN_2006-04_2021-09']['G1']; df_ = out['MAIN_2006-04_2021-09']['diff']
c_self = g1['mean_net_pct'] > 0 and (g1['t_month'] or -9) >= 2 and (g1['mean_net_ex_top5pct'] or -9) > 0
c_diff = (df_.get('month_clustered_diff_t') or -9) >= 2
verdict = '可用' if c_self and c_diff else ('待觀察' if c_self else '不可用')

yearly = {}
for y, e in MAIN.groupby(MAIN.date.dt.year):
    yearly[int(y)] = {g: ({'n': int((e.grp == g).sum()), 'net_pct': round(float(e[e.grp == g].net.mean() * 100), 2)} if (e.grp == g).any() else {'n': 0})
                      for g in ('G1', 'G2', 'G3')}
    yearly[int(y)]['bear_share'] = round(float(e.bear.mean()), 2)

grp_mix = {'share_by_group_main': MAIN.grp.value_counts(normalize=True).round(3).to_dict(),
           'G3_reason_main': {'not_top300': int((~MAIN.in_top300).sum()), 'top300_but_no_rev': int((MAIN.in_top300 & ~MAIN.has_rev).sum()),
                              'top300_rev_but_no_s52': int(((MAIN.grp == 'G3') & MAIN.in_top300 & MAIN.has_rev).sum())},
           'G1_rank_quantiles': MAIN[MAIN.grp == 'G1'].main_rank.quantile([.1, .5, .9]).round(1).to_dict(),
           'G1_s52_median': round(float(MAIN[MAIN.grp == 'G1'].s52.median()), 3), 'G2_s52_median': round(float(MAIN[MAIN.grp == 'G2'].s52.median()), 3),
           'G1_yoy3_median': round(float(MAIN[MAIN.grp == 'G1'].yoy3.median()), 1), 'G2_yoy3_median': round(float(MAIN[MAIN.grp == 'G2'].yoy3.median()), 1),
           'pool_n_median': int(MAIN.pool_n.median())}
top = MAIN[MAIN.grp == 'G1'].sort_values('ab', ascending=False).head(10)
top10_G1 = [{'code': r.code, 'date': str(r.date.date()), 'raw_pct': round(r.raw * 100, 1), 'ab_pct': round(r.ab * 100, 1), 'rank': r.main_rank, 'corp_flag': r.corp_flag} for r in top.itertuples()]

# ---- 組合：Sleeve，只收 G1；閒置 0050 ----
def mk_events(df, rt=0.008):
    ev = {}
    for r in df.itertuples():
        vals = [v for v in (r.v_in, r.v_out) if np.isfinite(v) and v > 0]
        ev.setdefault(r.i0, []).append((r.j, r.k, rt, 0.05 * min(vals) if vals else 0.0))
    return ev
E50 = L.E50
def run_sleeve(df, idle='0050', start='2006-04-01', end='2021-09-30', capital=1e6, rt=0.008):
    s = Dl.Sleeve(mk_events(df, rt), 10, idle, True)
    t0 = idx.searchsorted(pd.Timestamp(start)); t1 = idx.searchsorted(pd.Timestamp(end), 'right') - 1
    s.deposit(capital, t0 - 1); nav = [capital]; inv = []
    for t in range(t0, t1 + 1):
        s.step(t); e = s.equity(t); nav.append(e); inv.append(s.pos_value(t) / e)
    stt = {k: v for k, v in s.stat.items() if k != 'cap_fill_ratio'}
    cf = np.array(s.stat['cap_fill_ratio']); stt['share_trades_capacity_bound'] = round(float((cf < 0.999).mean()), 3) if len(cf) else None
    stt['mean_invested_share'] = round(float(np.mean(inv)), 3)
    tr = np.array([x[2] for x in s.trades]); stt['closed_trades'] = int(len(tr))
    if len(tr) > 2: stt['trade_net_mean_pct'] = round(float(tr.mean() * 100), 2); stt['trade_win'] = round(float((tr > 0).mean()), 3)
    return pd.Series(nav, index=idx[t0 - 1:t1 + 1]), stt
def pstats(nav):
    b = E50.reindex(nav.index)
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    o = {'CAGR': round(float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1), 4), 'maxDD': round(float((nav / nav.cummax() - 1).min()), 4),
         '0050_CAGR': round(float((b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1), 4), '0050_maxDD': round(float((b / b.cummax() - 1).min()), 4)}
    me = nav.groupby(nav.index.to_period('M')).last(); mb = b.groupby(b.index.to_period('M')).last()
    ex = (me.pct_change() - mb.pct_change()).dropna()
    o.update({'excess_vs_0050_pct_m': round(float(ex.mean() * 100), 2), 't_vs_0050_m': tstat(ex), 'n_months': int(len(ex)),
              't_vs_0050_daily': tstat(nav.pct_change() - b.pct_change())})
    o['yearly_pct'] = {}; o['0050_yearly_pct'] = {}
    for y in sorted(set(nav.index.year)):
        prev = nav[nav.index.year < y]; cur = nav[nav.index.year == y]
        if len(prev) == 0: continue
        o['yearly_pct'][int(y)] = round(float(cur.iloc[-1] / prev.iloc[-1] - 1) * 100, 1)
        o['0050_yearly_pct'][int(y)] = round(float(b[b.index.year == y].iloc[-1] / b[b.index.year < y].iloc[-1] - 1) * 100, 1)
    return o
port = {}; navs = {}
for name, df, kw in (('G1_only_0050', MAIN[MAIN.grp == 'G1'], {}), ('all_events_0050', MAIN, {}), ('G2uG3_0050', MAIN[MAIN.grp != 'G1'], {}),
                     ('G1_only_0050_cost1.2', MAIN[MAIN.grp == 'G1'], {'rt': 0.012}), ('G1_only_cash', MAIN[MAIN.grp == 'G1'], {'idle': 'cash'})):
    nav, stt = run_sleeve(df, **kw); navs[name] = nav
    port[name] = {'diag': stt, 'full_2006-04_2021-09': pstats(nav),
                  'sub_2006-04_2012-12': {k: v for k, v in pstats(nav[nav.index <= '2012-12-31']).items() if 'yearly' not in k},
                  'sub_2013-01_2021-09': {k: v for k, v in pstats(nav[nav.index >= nav[nav.index <= '2012-12-31'].index[-1]]).items() if 'yearly' not in k}}
pd.DataFrame(navs).to_parquet(HERE / 'nav.parquet')

# ---- 並列：2021～2026 真實名單（combo_search CROSS）----
cs = json.loads((ROOT / 'agents' / 'combo_search' / 'results.json').read_text(encoding='utf-8'))['CROSS']['b2_disposal_by_main_rank']

res.update({'coverage': cover, 'exec_stats_proxy_all_years': stP, 'groups': out, 'group_mix': grp_mix, 'yearly_main': yearly, 'top10_G1_main': top10_G1,
            'DECISION': {'G1_inputs': {k: g1[k] for k in ('n', 'mean_net_pct', 't_month', 'mean_net_ex_top5pct')},
                         'diff_inputs': df_, 'G1_self_ok': bool(c_self), 'diff_ok': bool(c_diff), 'verdict': verdict},
            'portfolio': port, 'REAL_2021_2026_combo_search_CROSS_b2': cs})
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')
EV.drop(columns=['src']).to_csv(HERE / 'events_grouped.csv.gz', index=False)
print(json.dumps(cover, ensure_ascii=False)); print(stP)
for k, v in out.items():
    print('==', k); [print('  ', g, v[g]) for g in ('G1', 'G2', 'G3', 'G2uG3', 'all')]; print('   diff', v['diff'])
print('DECISION', res['DECISION']); print(grp_mix); print(yearly); print(top10_G1)
for k, v in port.items(): print(k, v['diag'], {kk: vv for kk, vv in v['full_2006-04_2021-09'].items() if 'yearly' not in kk}, v['full_2006-04_2021-09']['yearly_pct'])
print('0050 yearly', port['G1_only_0050']['full_2006-04_2021-09']['0050_yearly_pct'])
