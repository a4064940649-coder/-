#!/usr/bin/env python3
"""把 results.json 轉成 FINDINGS 用的 markdown 表（只讀 results.json）。python3 -I -B tables.py"""
import json
from pathlib import Path
R = json.loads((Path(__file__).resolve().parent / 'results.json').read_text(encoding='utf-8'))
P = R['portfolio']; C = P['candidates']; RB = P['robust']
pc = lambda x, d=2: '—' if x is None else f'{x * 100:.{d}f}%'
pp = lambda x, d=2: '—' if x is None else f'{x * 100:+.{d}f}'
tt = lambda x: '—' if x is None else f'{x:.2f}'
NM = {'E1_hold20': 'E1 固定 20 日', 'E2_hold60': 'E2 固定 60 日', 'E3_stop8_ma50': 'E3 停損 8%＋破 50 日線（主）', 'E4_stop8_tp20': 'E4 停損 8%＋停利 20%',
      'E5_close_below_pivot': 'E5 收盤破樞紐', 'E6_trail10': 'E6 移動停損 10%'}
print('### 主欄（100 萬、最多 10 檔、一般成本、原始價訊號）')
print('| 出場規則 | 空位 | 2019 起年化 | 0050 同期 | 超額 pp | 月 t | 全期年化 | 0050 | 超額 pp | 月 t | 2006～2012 | 2013～2018 | 2019 起 | 最大回撤 | 拿掉最好 3 年 t | 平均投入 |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, o in C.items():
    r, idle = k.split('|')
    print(f"| {NM[r]} | {'現金' if idle == 'CASH' else '0050'} | **{pc(o['y2019_on']['cagr'])}** | {pc(o['y2019_on']['bench'])} | **{pp(o['y2019_on']['excess'])}** | {tt(o['y2019_on']['t'])} | {pc(o['full']['cagr'])} | {pc(o['full']['bench'])} | **{pp(o['full']['excess'])}** | {tt(o['full']['t'])} | {pp(o['p1_2006_2012']['excess'])} | {pp(o['p2_2013_2018']['excess'])} | {pp(o['y2019_on']['excess'])} | {pc(o['full']['maxdd'], 1)} | {tt(o['ex_best3']['t_ex_best3'])} | {o['info']['avg_invested'] * 100:.0f}% |")
print('\n### 挑選／驗證／保留')
print('| 出場規則 | 空位 | 挑選 2006～2014 超額（t） | 驗證 2015～2020 超額（t） | 保留 2021～2026 超額（t） | 2024 起超額（t） |')
print('|---|---|---|---|---|---|')
for k, o in C.items():
    r, idle = k.split('|')
    print(f"| {NM[r]} | {'現金' if idle == 'CASH' else '0050'} | {pp(o['sel_2006_2014']['excess'])}（{tt(o['sel_2006_2014']['t'])}） | {pp(o['val_2015_2020']['excess'])}（{tt(o['val_2015_2020']['t'])}） | {pp(o['hold_2021_2026']['excess'])}（{tt(o['hold_2021_2026']['t'])}） | {pp(o['y2024_on']['excess'])}（{tt(o['y2024_on']['t'])}） |")
print('\n### 交易統計（現金版）')
print('| 出場規則 | 筆數 | 平均淨報酬 | 中位 | 勝率 | 平均賺 | 平均賠 | 平均持有日 | 前 10 筆占總損益 | 買進周轉／年 | 成本拖累／年 | 出場原因 |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, o in C.items():
    r, idle = k.split('|')
    if idle != 'CASH': continue
    t = o['trades']
    print(f"| {NM[r]} | {t['n']} | {pc(t['mean_ret_net'])} | {pc(t['median_ret_net'])} | {pc(t['win'], 1)} | {pc(t['avg_win'], 1)} | {pc(t['avg_loss'], 1)} | {t['avg_hold_days']:.0f} | {pc(t['top10_pnl_share'], 0)} | {o['turnover_buy_per_year_x_nav']:.2f} 倍 | {pc(o['cost_drag_per_year_pct_of_avg_nav'])} | {t['reasons']} |")
print('\n### 穩健性（主規格 E3）')
print('| 版本 | 2019 起年化 | 超額 pp（t） | 全期年化 | 超額 pp（t） | 2006～2012 | 2013～2018 | 最大回撤 | 平均投入 | 買進筆數 | 5% 上限觸及 | 買不起 |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
def rr(lab, o):
    print(f"| {lab} | {pc(o['y2019_on']['cagr'])} | {pp(o['y2019_on']['excess'])}（{tt(o['y2019_on']['t'])}） | {pc(o['full']['cagr'])} | {pp(o['full']['excess'])}（{tt(o['full']['t'])}） | {pp(o['p1_2006_2012']['excess'])} | {pp(o['p2_2013_2018']['excess'])} | {pc(o['full']['maxdd'], 1)} | {o['info']['avg_invested'] * 100:.0f}% | {o['info']['n_buy']} | {o['info']['cap_bound']} | {o['info']['skip_lot']} |")
for idle, il in (('CASH', '現金'), ('P50', '0050')):
    rr(f'{il}｜100 萬一般（主）', C[f'E3_stop8_ma50|{idle}'])
    for k, lab in [('stress', '壓力成本'), ('K20', '最多 20 檔'), ('30萬', '30 萬'), ('30萬|stress', '30 萬壓力'), ('300萬', '300 萬'), ('300萬|stress', '300 萬壓力'), ('1000萬', '1,000 萬'), ('1000萬|stress', '1,000 萬壓力'),
                   ('lag1', '延後 1 日'), ('lag2', '延後 2 日'), ('lag3', '延後 3 日'), ('bars>=660', '≥ 660 根'), ('ADJ_signal', '還原價訊號')]:
        kk = f'E3_stop8_ma50|{idle}|{k}'
        if kk in RB: rr(f'{il}｜{lab}', RB[kk])
for k in RB:
    if k.startswith('E3_stop8_ma50|') and ('zero' in k or 'FIXED' in k or 'disposal' in k or 'open_limit' in k or 'TDR' in k): rr(k.replace('E3_stop8_ma50|', '').replace('|', '｜'), RB[k])
print('\n### 各出場規則的對照版（全期超額 pp／2019 起超額 pp）')
print('| 出場規則 | 現金·壓力 | 現金·20 檔 | 每筆固定 10 萬不限檔 | 現金·零成本（回測上緣，實現不了） | 0050·壓力 | 0050·20 檔 | 現金·還原價訊號 | 0050·還原價訊號 |')
print('|---|---|---|---|---|---|---|---|---|')
for r in NM:
    f = lambda k: f"{pp(RB[k]['full']['excess'])}／{pp(RB[k]['y2019_on']['excess'])}"
    print(f"| {NM[r]} | {f(r + '|CASH|stress')} | {f(r + '|CASH|K20')} | {f(r + '|FIXED100k_unlimited')} | {f(r + '|CASH|zero_cost(回測上緣，實現不了)')} | {f(r + '|P50|stress')} | {f(r + '|P50|K20')} | {f(r + '|CASH|ADJ_signal')} | {f(r + '|P50|ADJ_signal')} |")
E = R['event_study']
print('\n### 事件研究', E['exclusions'])
def ev(tag, S):
    for h in ('5', '10', '20', '60'):
        o = S[h]
        if o is None: continue
        xp = o['xpool']; x5 = o['x0050']
        print(f"| {tag} | {h} | {o['n']} | {pc(o['mean'])} | {pc(o['median'])} | {pc(o['win'], 1)} | {pc(o['mean_net'])} | {pc(o['bench0050_mean'])} | **{pp(x5['mean'])}**（{tt(x5['t'])}） | {pp(x5['median'])} | {pc(x5['win'], 1)} | {pp(o['x0050_net']['mean'])}（{tt(o['x0050_net']['t'])}） | {pc(o['pool_mean'])} | **{pp(xp['mean'])}**（{tt(xp['t'])}） | {pp(xp['median'])} | {pc(xp['win'], 1)} | {pp(o['xpool_net']['mean'])}（{tt(o['xpool_net']['t'])}） | {xp['G']} | {xp['neff']:.0f} | {tt(xp.get('t_month_mean'))} |")
print('| 樣本 | 持有日 | 筆數 | 平均（毛） | 中位 | 勝率 | 平均（淨） | 0050 同窗 | 對 0050 超額 pp（t） | 中位超額 | 贏 0050 比例 | 淨超額（t） | 同池平均 | 對同池超額 pp（t） | 中位超額 | 贏同池比例 | 淨超額（t） | 訊號月份數 | 有效樣本 | 月等權 t |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, lab in [('all', '全部'), ('sel_2006_2014', '挑選 2006～2014'), ('val_2015_2020', '驗證 2015～2020'), ('hold_2021_2026', '保留 2021～2026'), ('p1_2006_2012', '2006～2012'), ('p2_2013_2018', '2013～2018'), ('p3_2019_on', '2019 起')]:
    ev(lab, E[k])
for h in ('5', '10', '20', '60'):
    o = E['all'][h]; print(h, 'boot pool', o['xpool_boot3m'], '0050', o['x0050_boot3m'])
for k in ('next_open_gap_vs_ref', 'next_open_gap_buyable', 'pool_next_open_gap_mean', 'signal_day_return', 'breakout_margin_close_over_pivot'): print(k, E[k])
print('\n### 消融')
print('| 版本 | 訊號數 | 可成交 | 20 日平均 | 中位 | 對同池 pp（t） | 對 0050 pp（t） | 60 日平均 | 對同池 pp（t） | 對 0050 pp（t） | 訊號月份 | 組合·現金 全期超額（t） | 2019 起 | 平均投入 | 組合·0050 全期超額（t） | 2019 起（t） |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, o in R['ablation'].items():
    print(f"| {k} | {o['n']:,} | {o['n_buyable']:,} | {pc(o['h20']['mean'])} | {pc(o['h20']['median'])} | {pp(o['h20']['xpool'])}（{tt(o['h20']['xpool_t'])}） | {pp(o['h20']['x0050'])}（{tt(o['h20']['x0050_t'])}） | {pc(o['h60']['mean'])} | {pp(o['h60']['xpool'])}（{tt(o['h60']['xpool_t'])}） | {pp(o['h60']['x0050'])}（{tt(o['h60']['x0050_t'])}） | {o['h60']['months']} | {pp(o['CASH_full']['excess'])}（{tt(o['CASH_full']['t'])}） | {pp(o['CASH_y2019_on']['excess'])} | {o['CASH_avg_invested'] * 100:.0f}% | {pp(o['P50_full']['excess'])}（{tt(o['P50_full']['t'])}） | {pp(o['P50_y2019_on']['excess'])}（{tt(o['P50_y2019_on']['t'])}） |")
N = R['neighborhood']
print('\n### 鄰域', N['n_combos'])
print('| 指標 | 最小 | 10 分位 | 中位 | 90 分位 | 最大 | 為正比例 |'); print('|---|---|---|---|---|---|---|')
for k, v in N['summary'].items():
    if k == 'n': print(f"| 訊號數 | {v['min']:.0f} | {v['p10']:.0f} | {v['median']:.0f} | {v['p90']:.0f} | {v['max']:.0f} | |"); continue
    if 'p1_' in k or 'p2_' in k: continue
    f = (lambda x: f'{x * 100:+.2f}') if k.endswith('excess') else (lambda x: f'{x:.2f}')
    print(f"| {k} | {f(v['min'])} | {f(v['p10'])} | {f(v['median'])} | {f(v['p90'])} | {f(v['max'])} | {v['share_positive'] * 100:.0f}% |")
print(N['default_rank_pct'], N['sign_flip'])
for dim, d in N['one_dim'].items(): print(dim, {k: (v['n'], pp(v['CASH_full_excess']), pp(v['P50_full_excess']), pp(v['CASH_y2019_excess']), pp(v['P50_y2019_excess'])) for k, v in d.items()})
print('\n### 逐年'); Y = R['yearly_returns']
print('| 年 | E3·現金 | E3·0050 | 0050 | E3·現金−0050 | E3·0050−0050 |'); print('|---|---|---|---|---|---|')
for y in Y['0050']:
    a, b, e = Y['E3_stop8_ma50|CASH'].get(y), Y['E3_stop8_ma50|P50'].get(y), Y['0050'][y]
    print(f"| {y} | {pc(a, 1)} | {pc(b, 1)} | {pc(e, 1)} | {pp(a - e, 1)} | {pp(b - e, 1)} |")
print('\n### signals', {k: v for k, v in R['signals'].items() if k not in ('manual_check_5',)})
print('\n### gate', json.dumps({k: v for k, v in R['data_gate'].items() if k != 'lowvol'}, ensure_ascii=False))
print({k: v for k, v in R['data_gate']['lowvol'].items() if k != 'by_year'})
print('\n### adj', {k: v for k, v in R['adj_basis'].items() if k not in ('event_study', 'only_adj_examples', 'by_year_adj')})
for h in ('5', '10', '20', '60'):
    o = R['adj_basis']['event_study']['all'][h]; print(h, o['n'], pc(o['mean']), pc(o['median']), pc(o['win']), 'x0050', pp(o['x0050']['mean']), tt(o['x0050']['t']), 'xpool', pp(o['xpool']['mean']), tt(o['xpool']['t']))
print('\n### critique', json.dumps(R['critique'], ensure_ascii=False))
print('\n### snooping', json.dumps({k: v for k, v in R['snooping'].items() if k != 'monthly_sr'}, ensure_ascii=False))
print('\n### selection', json.dumps(R['selection_protocol'], ensure_ascii=False))
print('\n### vs_main', json.dumps(R['vs_main'], ensure_ascii=False))
print('\n### gates', json.dumps(R['gates'], ensure_ascii=False))
