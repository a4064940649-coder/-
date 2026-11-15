#!/usr/bin/env python3
"""S9 事後分析（只描述，不影響分類與判定）：只讀 results.json、navs.csv.gz、trades.csv.gz，不建面板。
Q1 逐年與 A10 的逐年名次；Q2 最大回撤高低點；Q3 哪幾筆交易造成差異；Q4 訊號日日曆；Q5（未登記）時效遞減的斜率；Q6 收盤成交的機制。
python3 -I -B posthoc.py → results_posthoc.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
R = json.loads((H / 'results.json').read_text())
NV = pd.read_csv(H / 'navs.csv.gz', index_col='date', parse_dates=['date'])
TR = pd.read_csv(H / 'trades.csv.gz', parse_dates=['sig_date', 'ent_date', 'exit_date'], dtype={'code': str})
FAM = ['A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A15']
ALL = FAM + ['B11opt', 'B11pes']
CELLS = ['W10_70', 'WN_70']
MODES = ['open', 'close']
out = {}


def yearly(s):
    y = s.groupby(s.index.year).last()
    first = s.iloc[0]
    r = {}
    prev = first
    for yr, v in y.items():
        r[int(yr)] = float(v / prev - 1); prev = v
    return r


# ---- Q1 逐年
q1 = {name: yearly(NV[name]) for name in NV.columns}
rank_by_year = {}
for cid in CELLS:
    for mode in MODES:
        rk = {}
        for yr in q1['0050']:
            vals = {d: q1[f'{d}|{mode}|{cid}'][yr] for d in FAM}
            rk[yr] = 1 + sum(1 for d, v in vals.items() if d != 'A10' and v > vals['A10'])
        rank_by_year[f'{cid}|{mode}'] = rk
out['Q1_yearly'] = q1
out['Q1_A10_rank_by_year'] = rank_by_year
print('Q1 A10 逐年名次（1＝最好，共 7）')
for k, v in rank_by_year.items():
    print(' ', k, v)
for cid in CELLS:
    print(' ', cid, 'open 逐年（%）')
    for d in ALL:
        print('   ', d.ljust(7), ' '.join(f"{q1[f'{d}|open|{cid}'][y]*100:6.1f}" for y in q1['0050']))
    print('    0050   ', ' '.join(f"{q1['0050'][y]*100:6.1f}" for y in q1['0050']))

# ---- Q2 最大回撤高低點
q2 = {}
for key, C in R['combos'].items():
    main = C['dd_dates']; pth = C['paths_dd_dates']
    ok_main = main['peak'][:7] == '2024-07' and main['trough'][:7] == '2025-04'
    ok_p = {k: (v[0][:7] == '2024-07' and v[1][:7] == '2025-04') for k, v in pth.items()}
    q2[key] = dict(main=[main['peak'], main['trough']], main_is_2024_07_to_2025_04=bool(ok_main), n_paths_same=int(sum(ok_p.values())),
                   others={k: pth[k] for k, v in ok_p.items() if not v})
n_main = sum(v['main_is_2024_07_to_2025_04'] for v in q2.values()); n_p = sum(v['n_paths_same'] for v in q2.values())
out['Q2_dd_dates'] = dict(per_combo=q2, n_main_same=n_main, n_main=len(q2), n_paths_same=n_p, n_paths=20 * len(q2))
print(f'Q2 主跑 {n_main}/{len(q2)}、路徑 {n_p}/{20*len(q2)} 的最大回撤是 2024-07 → 2025-04')
for k, v in q2.items():
    if not v['main_is_2024_07_to_2025_04'] or v['n_paths_same'] < 20:
        oth = {}
        for pk, (a, b) in v['others'].items():
            oth.setdefault(f'{a[:7]}→{b[:7]}', 0); oth[f'{a[:7]}→{b[:7]}'] += 1
        print('  例外', k, v['main'], f"路徑 {v['n_paths_same']}/20", oth)


# ---- Q3 哪幾筆交易造成差異（依股票加總損益的差，元；100 萬主跑）
def by_code(run):
    x = TR[(TR['run'] == run)]
    return x.groupby('code')['pnl'].sum()


q3 = {}
pairs = [(f'A10|open|{c}', f'{d}|open|{c}') for c in CELLS for d in FAM + ['B11opt'] if d != 'A10'] + [(f'A10|open|{c}', f'A10|close|{c}') for c in CELLS]
for a, b in pairs:
    pa, pb = by_code(a), by_code(b)
    df = pd.concat([pa.rename('a'), pb.rename('b')], axis=1).fillna(0.0)
    df['diff'] = df['a'] - df['b']
    top = df.sort_values('diff', ascending=False)
    tot = float(df['diff'].sum())
    q3[f'{a} − {b}'] = dict(total_pnl_diff=tot, top_pos=[(i, round(r['diff'])) for i, r in top.head(5).iterrows()],
                           top_neg=[(i, round(r['diff'])) for i, r in top.tail(5).iloc[::-1].iterrows()],
                           top5_share=float(top.head(5)['diff'].sum() / tot) if tot else None,
                           n_codes_only_a=int(((df['a'] != 0) & (df['b'] == 0)).sum()), n_codes_only_b=int(((df['b'] != 0) & (df['a'] == 0)).sum()))
out['Q3_trade_diff'] = q3
print('Q3 依股票加總損益差（A10 − 其他，元）')
for k, v in q3.items():
    print(' ', k, f"總差 {v['total_pnl_diff']/1e4:.0f} 萬；前 5 檔占 {v['top5_share']*100 if v['top5_share'] else float('nan'):.0f}%", '正', v['top_pos'], '負', v['top_neg'])
# 最賺幾檔在各定義的進出日
key_codes = ['2408', '2603', '1325', '3653', '2059', '6446']
detail = {}
for cid in CELLS:
    for code in key_codes:
        rows = []
        for d in ALL:
            x = TR[(TR['run'] == f'{d}|open|{cid}') & (TR['code'] == code)]
            for r in x.itertuples():
                if r.pnl > 1e5 or r.ret > 0.5:
                    rows.append((d, str(r.ent_date.date()), str(r.exit_date.date()), round(r.ret, 3)))
        detail[f'{cid}|{code}'] = rows
out['Q3_key_codes'] = detail

# ---- Q4 訊號日日曆（訊號日 k；open 在 k＋1 開盤成交）
cal = NV.index
pos = {d: i for i, d in enumerate(cal)}
q4 = {}
base = R['defs']['A10']['exec_dates']
for d in ALL:
    ex = R['defs'][d]['exec_dates']
    lags = []; same = 0; n = 0
    for m, dt in ex.items():
        if dt is None or base.get(m) is None or m < '2016-01':
            continue
        a, b = pd.Timestamp(dt), pd.Timestamp(base[m])
        if a not in pos or b not in pos:
            continue
        n += 1; L = pos[a] - pos[b]; lags.append(L); same += int(L == 0)
    q4[d] = dict(n_months=n, n_same_as_A10=same, mean_lag_td=float(np.mean(lags)), lag_dist={int(k): int(v) for k, v in zip(*np.unique(lags, return_counts=True))},
                 n_stale_months=R['defs'][d]['n_stale_months'], n_early_months=len([m for m in R['defs'][d]['early_months'] if m >= '2016-01']))
out['Q4_calendar'] = q4
print('Q4 訊號日相對 A10（交易日）')
for d, v in q4.items():
    print(' ', d, v)

# ---- Q5（事後、未登記）時效遞減：訊號日平均晚幾個交易日 vs 報酬（營收資訊相同的 A10～A15＋B11opt，open）
q5 = {}
for cid in CELLS:
    ds = ['B11opt', 'A10', 'A11', 'A12', 'A13', 'A15']
    x = np.array([q4[d]['mean_lag_td'] for d in ds])
    y1 = np.array([R['combos'][f'{d}|open|{cid}']['windows']['since2019']['cagr'] * 100 for d in ds])
    y2 = np.array([R['combos'][f'{d}|open|{cid}']['paths20']['median'] * 100 for d in ds])
    b1 = np.polyfit(x, y1, 1); b2 = np.polyfit(x, y2, 1)
    res1 = y1 - np.polyval(b1, x)
    q5[cid] = dict(defs=ds, mean_lag=x.tolist(), r19=y1.tolist(), paths_median=y2.tolist(), slope_r19_pp_per_day=float(b1[0]), slope_paths_pp_per_day=float(b2[0]),
                   A10_residual_pp=float(res1[1]), resid_sd_pp=float(res1.std(ddof=2)))
out['Q5_decay_posthoc_unregistered'] = q5
print('Q5（事後、未登記）每晚 1 個交易日的報酬變化')
for c, v in q5.items():
    print(' ', c, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()})

# ---- Q6 收盤成交的機制
q6 = {}
for cid in CELLS:
    for d in ALL:
        o = R['combos'][f'{d}|open|{cid}']; c = R['combos'][f'{d}|close|{cid}']
        q6[f'{d}|{cid}'] = dict(blk_limup_open=o['blk_limup'], blk_limup_close=c['blk_limup'], defer_limdn_open=o['sell_defer_limdn'], defer_limdn_close=c['sell_defer_limdn'],
                                n_buy_open=o['n_buy'], n_buy_close=c['n_buy'],
                                ev_open=o['trades_full']['expectancy'], ev_close=c['trades_full']['expectancy'],
                                n_open=o['trades_full']['n'], n_close=c['trades_full']['n'])
out['Q6_close_mechanism'] = q6
print('Q6 漲停買不到／跌停賣不掉、每筆期望值（open → close）')
for k, v in q6.items():
    print(' ', k, f"漲停買不到 {v['blk_limup_open']}→{v['blk_limup_close']}；跌停延後賣 {v['defer_limdn_open']}→{v['defer_limdn_close']}；買進 {v['n_buy_open']}→{v['n_buy_close']}；"
          f"期望值 {v['ev_open']*100:.2f}%→{v['ev_close']*100:.2f}%")
# A10 close 回撤解剖：回撤期間兩版差在哪
for cid in CELLS:
    a, b = NV[f'A10|open|{cid}'], NV[f'A10|close|{cid}']
    seg = {}
    for lo, hi in [('2024-07-01', '2024-08-31'), ('2024-09-01', '2024-12-31'), ('2025-01-01', '2025-03-31'), ('2025-04-01', '2025-04-09')]:
        sa = a.loc[:lo].iloc[-2:]; sb = b.loc[:lo].iloc[-2:]
        seg[f'{lo}~{hi}'] = dict(open=float(a.loc[:hi].iloc[-1] / a.loc[:lo].iloc[-1] - 1), close=float(b.loc[:hi].iloc[-1] / b.loc[:lo].iloc[-1] - 1))
    out.setdefault('Q6_dd_segments', {})[cid] = seg
    print(' ', cid, 'A10 open vs close 分段報酬', {k: (round(v['open'] * 100, 1), round(v['close'] * 100, 1)) for k, v in seg.items()})

(H / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print('寫入 results_posthoc.json')
