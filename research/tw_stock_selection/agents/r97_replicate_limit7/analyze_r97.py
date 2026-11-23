#!/usr/bin/env python3
"""R97 分析：交易差異、抽查（原始日 K 手算漲跌停）、以資料驗證制度切換日、其他檢查。讀 out_*.json／trades_*.csv.gz（run_r97.py 產出）＋ raw/pre2015 原始檔。
python3 -I -B analyze_r97.py → analysis.json、spot_check.csv"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RS = HERE.parent.parent
CUT = pd.Timestamp('2015-06-01')
A = {}
d = pd.read_csv(RS / 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz', dtype={'code': str}, parse_dates=['date'])
d = d[d['close'] > 0]
# 2015-01-05 起引擎用官方面板（快取 panel_cs.npz）；有些代號（如 4426）原始檔 2015 年沒有 → 2015-01-05 起一律改取官方面板的原始開高低收，xflag 沿用原始檔（沒有就 -1）
SEAM = pd.Timestamp('2015-01-05')
_z = np.load('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s75/cache_cont/panel_cs.npz', allow_pickle=False)
_idx = pd.DatetimeIndex(_z['IDX']); _rows = np.flatnonzero((_idx >= SEAM) & (_idx <= pd.Timestamp('2015-06-30'))); _cols = np.array(_z['COLS']); _mk = _z['MK'][_rows] == 1
_fr = {}
for _k, _n in (('O', 'open'), ('H', 'high'), ('L', 'low'), ('C', 'close'), ('VOL', 'volume'), ('AMT', 'amount'), ('CHG', 'change')):
    _a = _z[_k][_rows]; _fr[_n] = _a[_mk]
_ti, _tj = np.nonzero(_mk)
off = pd.DataFrame(dict(date=_idx[_rows][_ti], code=_cols[_tj], **_fr)); off = off[off['close'] > 0]
off = off.merge(d.loc[d['date'] >= SEAM, ['date', 'code', 'xflag']], on=['date', 'code'], how='left'); off['xflag'] = off['xflag'].fillna(-1).astype(int)
d = pd.concat([d[d['date'] < SEAM].assign(src='raw/pre2015'), off.assign(src='official_panel')], ignore_index=True)[['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'xflag', 'src']]
d = d.sort_values(['code', 'date']).reset_index(drop=True)
del _z, off, _fr
g = d.groupby('code')
d['prev_close'] = g['close'].shift(1); d['prev_date'] = g['date'].shift(1); d['next_open'] = g['open'].shift(-1); d['next_date'] = g['date'].shift(-1)
d['ref_from_change'] = d['close'] - d['change'].fillna(0)


def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0


def lim(ref, pct):
    u = ref * (1 + pct); tu = tick(u); u = math.floor(u / tu + 1e-9) * tu
    l = ref * (1 - pct); tl = tick(l); l = math.ceil(l / tl - 1e-9) * tl
    return round(u, 2), round(l, 2)


# ---------------- (C) 用資料驗證制度：每天全市場「收盤對前收」最大漲幅／最大跌幅（排除除權息日 xflag=1）
x = d[(d['xflag'] == 0) & d['prev_close'].notna() & ((d['change'].fillna(0) != 0) | (d['close'] == d['prev_close']))].copy()
x['r'] = x['close'] / x['prev_close'] - 1; x['rl'] = x['low'] / x['prev_close'] - 1; x['rh'] = x['high'] / x['prev_close'] - 1
x = x[(x['r'].abs() < 0.2)]
dm = x.groupby('date').agg(up=('rh', lambda s: s.quantile(0.999)), dn=('rl', lambda s: s.quantile(0.001)))
pre, post = dm[dm.index < CUT], dm[dm.index >= CUT]
A['regime_from_data'] = dict(
    說明='原始日 K（raw/pre2015）每天全市場 最高÷前收−1 的 99.9 百分位、最低÷前收−1 的 0.1 百分位（排除除權息日）',
    pre_2015_06_01=dict(days=int(len(pre)), max_up=float(pre['up'].max()), days_up_over_7_1pct=int((pre['up'] > 0.071).sum()), min_dn=float(pre['dn'].min()), days_dn_below_7_1pct=int((pre['dn'] < -0.071).sum())),
    from_2015_06_01_to_06_30=dict(days=int(len(post)), max_up=float(post['up'].max()), first_day_up_over_7_1pct=str(post.index[(post['up'] > 0.071)][0].date()) if (post['up'] > 0.071).any() else None, min_dn=float(post['dn'].min())),
    last_week_before=[(str(i.date()), round(float(r['up']), 4), round(float(r['dn']), 4)) for i, r in dm.loc['2015-05-25':'2015-06-05'].iterrows()])
# 2008-10 跌幅減半（3.5%）期間：每天最低÷前收−1 的全市場最小值（0.1 百分位）
w = dm.loc['2008-10-06':'2008-11-05']
A['oct2008_halved_down_limit'] = [(str(i.date()), round(float(r['up']), 4), round(float(r['dn']), 4)) for i, r in w.iterrows()]
half = dm[(dm['dn'] > -0.04) & (dm.index >= '2008-09-01') & (dm.index <= '2008-12-31')]
A['oct2008_days_min_low_above_minus4pct'] = [str(i.date()) for i in half.index]

# ---------------- (A) 交易差異
tb = pd.read_csv(HERE / 'trades_base.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
tl = pd.read_csv(HERE / 'trades_lim7.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
m = tb.merge(tl, on=['code', 'ent_date'], how='outer', suffixes=('_b', '_l'), indicator=True)
only_b = m[m['_merge'] == 'left_only']; only_l = m[m['_merge'] == 'right_only']; both = m[m['_merge'] == 'both']
chg = both[(both['shares_b'] != both['shares_l']) | (both['exit_date_b'] != both['exit_date_l'])]
f = lambda r, s: dict(code=r['code'], ent_date=str(r['ent_date'].date()), exit_date=str(r[f'exit_date{s}'].date()), shares=int(r[f'shares{s}']), cost=round(float(r[f'cost{s}'])), ret_pct=round(float(r[f'ret{s}']) * 100, 2), pnl=round(float(r[f'pnl{s}'])), why=r[f'why{s}'])
A['trade_diff'] = dict(n_base=int(len(tb)), n_lim7=int(len(tl)), n_same_key=int(len(both)),
                       only_in_base=[f(r, '_b') for _, r in only_b.sort_values('ent_date').iterrows()],
                       only_in_lim7=[f(r, '_l') for _, r in only_l.sort_values('ent_date').iterrows()],
                       same_key_diff_shares_or_exit=dict(n=int(len(chg)), n_pre_cut=int((chg['ent_date'] < CUT).sum()), n_exit_date_differs=int((chg['exit_date_b'] != chg['exit_date_l']).sum()),
                                                         first=str(chg['ent_date'].min().date()) if len(chg) else None,
                                                         pre_cut_examples=[dict(code=r['code'], ent_date=str(r['ent_date'].date()), shares_base=int(r['shares_b']), shares_lim7=int(r['shares_l']))
                                                                           for _, r in chg[chg['ent_date'] < CUT].sort_values('ent_date').head(40).iterrows()]),
                       same_key_identical=int(len(both) - len(chg)))
ob = json.loads((HERE / 'out_base.json').read_text()); ol = json.loads((HERE / 'out_lim7.json').read_text())
locked = ol['list_pre_cut']['locked']
bk = {(r['code'], str(r['ent_date'].date())): r for _, r in tb.iterrows()}
lk = {(r['code'], str(r['ent_date'].date())): r for _, r in tl.iterrows()}
# ---------------- (B) 7 筆全列＋抽查（原始日 K 手算）
rows = []
for r in locked:
    q = d[(d['code'] == r['code']) & (d['date'] == r['date'])].iloc[0]
    u7, l7 = lim(q['prev_close'], 0.07); u10, l10 = lim(q['prev_close'], 0.10)
    b = bk.get((r['code'], r['date'])); l_ = lk.get((r['code'], r['date']))
    rows.append(dict(code=r['code'], date=r['date'], ohlc_source=q['src'], list_month=r['month'], prev_date=str(q['prev_date'].date()), prev_close=q['prev_close'], xflag=int(q['xflag']), raw_change=q['change'],
                     open=q['open'], high=q['high'], low=q['low'], close=q['close'], volume=int(q['volume']),
                     tick_at_7pct=tick(q['prev_close'] * 1.07), raw_7pct=round(q['prev_close'] * 1.07, 4), limup7_hand=u7, limdn7_hand=l7, tick_at_10pct=tick(q['prev_close'] * 1.10), limup10_hand=u10, limdn10_hand=l10,
                     engine_ref=r['ref'], engine_limup7=round(r['limup7'], 2), engine_limup10=round(r['limup10'], 2), ref_matches_prev_close=bool(abs(r['ref'] - q['prev_close']) < 1e-6),
                     open_ge_limup7=bool(q['open'] >= u7 - 1e-9), open_ge_limup10=bool(q['open'] >= u10 - 1e-9), locked_all_day=bool(q['low'] >= u7 - 1e-9), close_at_limup7=bool(q['close'] >= u7 - 1e-9),
                     next_date=str(q['next_date'].date()), next_open=q['next_open'], next_open_vs_open_pct=round((q['next_open'] / q['open'] - 1) * 100, 2),
                     base_handling=(f"買進 {int(b['shares'])} 股，成本 {b['cost']:.0f}，{str(b['exit_date'].date())} 出場（{b['why']}），報酬 {b['ret'] * 100:.2f}%，損益 {b['pnl']:.0f}" if b is not None else '未買（其他原因：整張／資金門檻等）'),
                     base_ret_pct=round(float(b['ret']) * 100, 2) if b is not None else None, base_pnl=round(float(b['pnl'])) if b is not None else None,
                     lim7_handling=('開盤價＝7% 漲停價 → 不買、不遞補' if l_ is None else '仍買進（異常）'),
                     spot_checked=None))
S = pd.DataFrame(rows)
pick = ['2478', '1516', '3406', '8201', '4426']             # 抽 5 筆：跨 3 個跳動點級距（0.05、0.1、0.5）、含盤中打開的 8201、含 2008-10 與 2015 的各一筆
S['spot_checked'] = S['code'].isin(pick)
S.to_csv(HERE / 'spot_check.csv', index=False)
A['locked7'] = rows
A['locked7_summary'] = dict(n=len(rows), n_bought_in_base=int(sum(r['base_ret_pct'] is not None for r in rows)), base_pnl_sum=int(sum(r['base_pnl'] or 0 for r in rows)),
                            base_ret_mean_pct=float(np.mean([r['base_ret_pct'] for r in rows if r['base_ret_pct'] is not None])), n_locked_all_day=int(sum(r['locked_all_day'] for r in rows)),
                            all_hand_equals_engine=bool(all(abs(r['limup7_hand'] - r['engine_limup7']) < 1e-6 and abs(r['limup10_hand'] - r['engine_limup10']) < 1e-6 and r['ref_matches_prev_close'] for r in rows)))
# ---------------- (D) 2015-06-01 前所有成交（lim7 情境）：開盤是否落在手算 7% 漲跌停之內；是否落在除權息日（引擎 REF＝當日收盤）
key = d.set_index(['code', 'date'])
idx_dates = pd.DatetimeIndex(sorted(d['date'].unique()))
def scan(out, name):
    fl = pd.DataFrame(out['fills_pre_cut'], columns=['t', 'code', 'side', 'open'])
    nav = pd.read_csv(HERE / f'nav_{name}.csv.gz', index_col=0, parse_dates=True)
    all_idx = pd.read_csv(HERE.parent / 's75_continuous_2008_2026' / 'v2_2026-11-21' / 'nav_2008_2026_v2.csv.gz', index_col=0, parse_dates=True).index
    res = dict(n=int(len(fl)), n_buy=int((fl['side'] == 'B').sum()), n_sell=int((fl['side'] == 'S').sum()), on_xflag_day=[], buy_open_ge_up7=[], sell_open_le_dn7=[], outside_7pct_band=[], no_prev=[], sell_open_le_dn3_5_in_oct2008=[], open_mismatch=0)
    return fl, res
import numpy as _np
z = _np.load('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s75/cache_cont/panel_cs.npz', allow_pickle=False)
PIDX = pd.DatetimeIndex(z['IDX'])
for name, out in (('base', ob), ('lim7', ol)):
    fl, res = scan(out, name)
    for t, c, s, o in fl.itertuples(index=False):
        dt = PIDX[t]
        if (c, dt) not in key.index:
            res['no_prev'].append((c, str(dt.date()), s)); continue
        q = key.loc[(c, dt)]
        if abs(q['open'] - o) > 1e-6: res['open_mismatch'] += 1
        if not np.isfinite(q['prev_close']):
            res['no_prev'].append((c, str(dt.date()), s)); continue
        u7, l7 = lim(q['prev_close'], 0.07)
        tag = (c, str(dt.date()), s, float(o), float(q['prev_close']), u7, l7)
        if q['xflag'] == 1 or abs((q['close'] - (q['change'] if np.isfinite(q['change']) else 0)) - q['prev_close']) > 1e-6: res['on_xflag_day'].append(tag + (int(q['xflag']), float(q['close']), float(q['change']) if np.isfinite(q['change']) else None))
        if s == 'B' and o >= u7 - 1e-9: res['buy_open_ge_up7'].append(tag)
        if s == 'S' and o <= l7 + 1e-9: res['sell_open_le_dn7'].append(tag)
        if (o > u7 + 1e-9 or o < l7 - 1e-9): res['outside_7pct_band'].append(tag)
        if s == 'S' and pd.Timestamp('2008-10-13') <= dt <= pd.Timestamp('2008-10-24'):
            l35 = q['prev_close'] * 0.965; tk = tick(l35); l35 = math.ceil(l35 / tk - 1e-9) * tk
            res['sells_in_oct2008_window'] = res.get('sells_in_oct2008_window', 0) + 1
            if o <= l35 + 1e-9: res['sell_open_le_dn3_5_in_oct2008'].append(tag + (round(l35, 2),))
    A[f'fills_scan_{name}'] = res
(HERE / 'analysis.json').write_text(json.dumps(A, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in A.items() if k != 'locked7'}, ensure_ascii=False, indent=1, default=str)[:9000])
print(S[['code', 'date', 'prev_close', 'open', 'high', 'low', 'close', 'limup7_hand', 'limup10_hand', 'locked_all_day', 'next_open', 'base_handling', 'spot_checked']].to_string())
