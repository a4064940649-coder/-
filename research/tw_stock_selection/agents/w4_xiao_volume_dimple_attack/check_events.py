#!/usr/bin/env python3
"""事後診斷與抽樣核對（不影響判定）：獨立重算訊號、手算事件、拆解、src 3 量價抽查。輸出寫入 results.json 的 diagnostics_posthoc。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402

P = R.P
ev = pd.read_csv(Path(P).parent / 'w4_xiao_PRIMARY_events.csv.gz', dtype={'code': str}, parse_dates=['sig', 'entry'])
out = {}

# 1) 獨立重算訊號：直接讀原始 parquet，不用 run.py 的中間變數
adj = pd.read_parquet(P / 'adj.parquet'); cr = pd.read_parquet(P / 'close_raw.parquet'); hi = pd.read_parquet(P / 'high.parquet')
op = pd.read_parquet(P / 'open.parquet'); val = pd.read_parquet(P / 'value.parquet'); src = pd.read_parquet(P / 'src.parquet')
keep = adj.notna().sum(axis=1) >= 300
adj, cr, hi, op, val, src = [x.loc[keep] for x in (adj, cr, hi, op, val, src)]
def sig_ok(c, d):
    i = adj.index.get_loc(d)
    a = adj[c].iloc[:i + 1]; v = (val[c] / cr[c]).where(val[c] > 0).iloc[:i + 1]
    pv = v.iloc[-61:-1]; v20 = pv.iloc[-20:].mean(); v60 = pv.mean(); p5 = pv.iloc[-5:]
    pa = a.iloc[-21:-1]
    v60val = val[c].iloc[:i + 1].fillna(0).iloc[-60:].mean()
    return dict(c1=bool((p5 < v20).all()), c2=bool(p5.mean() <= 0.6 * v60), c3=bool(pa.max() / pa.min() - 1 <= 0.15),
                c4=bool(v.iloc[-1] > p5.max()), c5=bool(v.iloc[-1] >= 1.5 * v20), c6=bool(a.iloc[-1] > a.iloc[-2]),
                c7=bool(a.iloc[-1] >= a.iloc[-20:].mean()), univ=bool(v60val >= 1e7 and cr.at[d, c] >= 10), nonan=bool(pv.notna().all()))
smp = ev.sample(200, random_state=1)
bad = []
for _, r in smp.iterrows():
    o = sig_ok(r['code'], r['sig'])
    if not all(o.values()): bad.append((r['code'], str(r['sig'].date()), o))
out['recheck_signal_sample200_failures'] = len(bad)
out['recheck_signal_failures_examples'] = [str(b) for b in bad[:5]]
print('signal recheck failures', len(bad), bad[:3])

# 2) 手算事件（含 vol_nohigh 與 ma20 出場各數筆）
def hand(r):
    c = r['code']; a = adj[c]; fac = adj[c] / cr[c]; o_adj = op[c] * fac; h_adj = hi[c] * fac
    v = (val[c] / cr[c]).where(val[c] > 0); vmax = v.rolling(20, min_periods=20).max()
    ma20 = a.rolling(20, min_periods=20).mean()
    ie = a.index.get_loc(r['entry']); p0 = o_adj.iloc[ie]
    ref = None; xs = None; why = 'time'
    for s in range(ie, ie + 60):
        if ref is not None and s > ref[0] and a.iloc[s] > ref[1]: ref = None
        if v.iloc[s] >= vmax.iloc[s]: ref = (s, h_adj.iloc[s])
        if a.iloc[s] <= 0.9 * p0: xs = s + 1; why = 'stop'; break
        if a.iloc[s] < ma20.iloc[s]: xs = s + 1; why = 'ma20'; break
        if ref is not None and s >= ref[0] + 5: xs = s + 1; why = 'vol_nohigh'; break
    if xs is None: xs = ie + 60
    px = o_adj.iloc[xs:].dropna().iloc[0]
    return {'code': c, 'sig': str(r['sig'].date()), 'entry': str(r['entry'].date()), 'p0': round(float(p0), 3), 'exit_day': str(a.index[xs].date()),
            'why_hand': why, 'why_run': r['why'], 'raw_hand': round(float(px / p0 - 1), 5), 'raw_run': round(float(r['raw']), 5)}
traces = [hand(r) for _, r in pd.concat([ev[ev['why'] == 'vol_nohigh'].sample(3, random_state=3), ev[ev['why'] == 'ma20'].sample(2, random_state=4),
                                         ev[ev['why'] == 'stop'].sample(1, random_state=5)]).iterrows()]
mism = 0
for _, r in ev.sample(300, random_state=11).iterrows():
    h = hand(r)
    if h['why_hand'] != h['why_run'] or abs(h['raw_hand'] - h['raw_run']) > 1e-6: mism += 1
out['hand_traces'] = traces; out['hand_recheck_300_mismatch'] = mism
print(json.dumps(traces, ensure_ascii=False, indent=0)); print('hand recheck 300 mismatch', mism)

# 3) 拆解
for w, (a_, b_) in R.WIN.items():
    d = ev[(ev['sig'] >= a_) & (ev['sig'] <= b_)]
    o = {'n': len(d), 'hold_dist': {str(k): round(float(v), 3) for k, v in d['hold'].clip(upper=21).value_counts(normalize=True).sort_index().items() if k in (1, 2, 3, 4, 5, 6, 10, 21)},
         'share_hold_le5': round(float((d['hold'] <= 5).mean()), 3),
         'mean_gap_close_to_open_pct': round(d['gap'].mean() * 100, 3),
         'mean_gross_ab_pct': round(d['ab_gross'].mean() * 100, 3), 'mean_cost_pct': round((d['raw'] - d['net']).mean() * 100, 3),
         'gross_ab_if_bought_at_signal_close_pct(unattainable)': round(((1 + d['raw']) * (1 + d['gap']) - 1 - d['ew']).mean() * 100, 3),
         'mean_raw_by_exit': {k: round(g['raw'].mean() * 100, 2) for k, g in d.groupby('why')},
         'mean_hold_by_exit': {k: round(g['hold'].mean(), 1) for k, g in d.groupby('why')},
         'top1pct_contrib_raw': round(float(d['raw'].nlargest(max(1, len(d) // 100)).sum() / len(d) * 100), 3)}
    out[f'decomp_{w}'] = o
    print(w, o)
out['yearly_ab_net_pct'] = {int(y): round(g['ab'].mean() * 100, 2) for y, g in ev.groupby(ev['sig'].dt.year)}
out['yearly_ab_gross_pct'] = {int(y): round(g['ab_gross'].mean() * 100, 2) for y, g in ev.groupby(ev['sig'].dt.year)}
out['yearly_n'] = {int(y): int(len(g)) for y, g in ev.groupby(ev['sig'].dt.year)}
print('yearly net', out['yearly_ab_net_pct']); print('yearly gross', out['yearly_ab_gross_pct']); print('yearly n', out['yearly_n'])

# 4) src 3 量價抽查
S3 = {}
dt = pd.read_csv('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/scr_extra/daytrade_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
dt = dt[(dt['dt_ratio'] >= 5) & (dt['dt_vol'] >= 200)]
dt['off_vol'] = dt['dt_vol'] / (dt['dt_ratio'] / 100) * 1000          # 官方總成交股數（當沖量 ÷ 當沖比率）
pv = (val / cr).where(val > 0).stack().rename('panel_vol').reset_index(); pv.columns = ['date', 'code', 'panel_vol']
ps = src.stack().rename('src').reset_index(); ps.columns = ['date', 'code', 'src']
m = dt.merge(pv, on=['date', 'code']).merge(ps, on=['date', 'code'])
m['ratio'] = m['panel_vol'] / m['off_vol']
g = m.groupby('src')['ratio']
S3['panel_vol_over_official_vol_by_src'] = {int(k): {'n': int(len(x)), 'median': round(float(x.median()), 4), 'p10': round(float(x.quantile(.1)), 4),
                                                    'p90': round(float(x.quantile(.9)), 4)} for k, x in g}
m['ym'] = m['date'].dt.to_period('Q').astype(str)
S3['median_ratio_by_quarter'] = {k: round(float(x.median()), 4) for k, x in m.groupby('ym')['ratio']}
print('src3 vol vs official', S3['panel_vol_over_official_vol_by_src']); print(S3['median_ratio_by_quarter'])
# 官方 history.csv.gz（2025-09-24 起）與面板比對
hcsv = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/history.csv.gz', dtype={'code': str}, parse_dates=['date'])
hh = hcsv.merge(pv, on=['date', 'code'])
hh = hh[hh['volume'] > 0]
S3['panel_vs_history_csv_src4'] = {'n': int(len(hh)), 'median_ratio': round(float((hh['panel_vol'] / hh['volume']).median()), 4),
                                   'share_within_1pct': round(float(((hh['panel_vol'] / hh['volume'] - 1).abs() < 0.01).mean()), 4)}
print('history.csv vs panel', S3['panel_vs_history_csv_src4'])
# 切換日前後的量跳動（截面中位數 log(後 10 日均量 / 前 10 日均量)），與安慰劑日比較
VOL = (val / cr).where(val > 0)
def jump(d0):
    i = VOL.index.searchsorted(pd.Timestamp(d0))
    a = VOL.iloc[i - 10:i].mean(); b = VOL.iloc[i:i + 10].mean()
    return round(float(np.log(b / a).median()), 4)
S3['xsec_median_log_vol_jump'] = {d: jump(d) for d in ['2024-02-15', '2025-09-24', '2023-09-25', '2024-09-24', '2025-08-25', '2026-03-24']}
print('jumps', S3['xsec_median_log_vol_jump'])
# 每年觸發數 / 股池股票日數
U = R.UNIV.sum(axis=1)
sig = pd.Series(R.signals(True).sum(axis=1), index=R.IDX)
yr = pd.DataFrame({'sig': sig, 'u': U}).groupby(R.IDX.year).sum()
S3['signal_rate_per_10k_univ_days'] = {int(k): round(float(v.sig / v.u * 1e4), 2) for k, v in yr.iterrows()}
srcd = pd.Series(np.where(R.SRC3, 3, 0), index=R.IDX)
S3['signal_rate_src3_vs_other_2021on'] = {
    'src3': round(float(sig[R.SRC3].sum() / U[R.SRC3].sum() * 1e4), 2),
    'other_2021on': round(float(sig[(~R.SRC3) & (R.IDX >= '2021-01-01')].sum() / U[(~R.SRC3) & (R.IDX >= '2021-01-01')].sum() * 1e4), 2),
    'first_20d_after_2025-09-24': round(float(sig['2025-09-24':].iloc[:20].sum() / U['2025-09-24':].iloc[:20].sum() * 1e4), 2)}
print('signal rates', S3['signal_rate_per_10k_univ_days'], S3['signal_rate_src3_vs_other_2021on'])
# 2025-05～07 成交值異常：每日截面中位數 panel/官方 比
dr = m.groupby('date')['ratio'].median()
badd = dr[dr > 1.5]
S3['days_ratio_gt_1.5'] = {'n': int(len(badd)), 'first': str(badd.index.min().date()), 'last': str(badd.index.max().date()),
                           'median_ratio_in_core_2025-05-08_2025-06-27': round(float(dr['2025-05-08':'2025-06-27'].median()), 3)}
tot = val.sum(axis=1) / 1e8
S3['market_value_1e8_median'] = {'2025-03-01~04-29': round(float(tot['2025-03-01':'2025-04-29'].median())), '2025-05-08~06-27': round(float(tot['2025-05-08':'2025-06-27'].median())),
                                 '2025-07-07~09-23': round(float(tot['2025-07-07':'2025-09-23'].median()))}
# 價格未受影響：2330 與 0050 日報酬相關
r5 = R.r50
rt = pd.read_parquet(P / 'ret.parquet')
S3['corr_2330_vs_0050'] = {f'{a}~{b}': round(float(rt.loc[a:b, '2330'].corr(r5.reindex(rt.loc[a:b].index))), 3)
                           for a, b in [('2023-03-01', '2023-09-30'), ('2025-05-02', '2025-07-04'), ('2025-07-07', '2025-09-23')]}
# 受影響訊號：訊號日 60 日量視窗含 2025-04-30～2025-07-04 → 訊號日 2025-04-30～2025-10-01
aff = (ev['sig'] >= '2025-04-30') & (ev['sig'] <= '2025-10-01')
vv = ev[(ev['sig'] >= '2019-05-01') & ~aff]; g2 = vv.groupby(vv['entry'].dt.to_period('M'))['ab'].mean()
S3['events_affected_by_value_bug'] = {'n': int(aff.sum()), 'mean_ab_net_pct': round(float(ev[aff]['ab'].mean() * 100), 3),
                                      'valid_ex_affected': {'n': int(len(vv)), 'mean_ab_net_pct': round(float(vv['ab'].mean() * 100), 3), 't_month': R.tstat(g2)},
                                      'monthly_n_2025': {str(k): int(v) for k, v in ev[(ev['sig'] >= '2025-01-01') & (ev['sig'] <= '2025-12-31')].groupby(ev['sig'].dt.to_period('M')).size().items()}}
print('value bug', {k: S3[k] for k in ('days_ratio_gt_1.5', 'market_value_1e8_median', 'corr_2330_vs_0050', 'events_affected_by_value_bug')})
out['src3_check'] = S3

f = HERE / 'results.json'
dd = json.loads(f.read_text(encoding='utf-8')); dd['diagnostics_posthoc'] = out
f.write_text(json.dumps(dd, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
