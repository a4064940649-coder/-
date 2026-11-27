# -*- coding: utf-8 -*-
"""R107 比對：my_numbers.json（獨立重算）vs results.json（原報告）。寫 compare.json。
容忍：比值類 |差| ≤ 0.01 或 相對差 ≤ 1%；金額類 相對差 ≤ 1%；蒙地卡羅 相對差 ≤ 2%；日期／判定 須完全相同。
執行：python3 -I -B compare.py（須先跑 replicate.py）
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, '..', 'eng_full_report_2026-11-27', 'results.json'))
M = json.load(open(os.path.join(HERE, 'my_numbers.json'), encoding='utf-8'))
R = json.load(open(RES, encoding='utf-8'))
mt, rt = M['tests'], R['tests']

rows = []


def add(item, mine, orig, kind='ratio', note=''):
    r = {'item': item, 'mine': mine, 'orig': orig, 'kind': kind, 'note': note}
    if kind in ('date', 'verdict', 'int'):
        r['diff'] = None
        r['ok'] = (mine == orig)
    else:
        d = mine - orig
        rel = abs(d) / abs(orig) if orig != 0 else (0.0 if d == 0 else float('inf'))
        r['diff'] = d
        r['rel_diff'] = rel
        if kind == 'ratio':
            r['ok'] = abs(d) <= 0.01 or rel <= 0.01
        elif kind == 'money':
            r['ok'] = rel <= 0.01
        elif kind == 'mc':
            r['ok'] = rel <= 0.02
        elif kind == 'ratio_strict':   # 占比數字本身很小，|差|≤0.01 太寬，只用相對 1%
            r['ok'] = rel <= 0.01
    rows.append(r)


# 1 複利
c, rc = M['compound'], R['compound']
add('複利 年化', c['cagr'] * 100, rc['cagr_pct'], note='單位 %')
add('複利 MDD', c['mdd'] * 100, rc['maxdd_pct'], note='單位 %')
add('複利 MDD 高點日', c['mdd_peak'], rc['peak'], 'date')
add('複利 MDD 低點日', c['mdd_trough'], rc['trough'], 'date')
add('複利 MDD 收復日', c['mdd_recover'], rc['recovered'], 'date')
add('複利 終值', c['final'], rc['final_nav'], 'money')
add('0050 年化', c['cagr_0050'] * 100, rc['cagr_0050_pct'], note='單位 %')
add('0050 終值', c['final_0050'], rc['final_nav_0050'], 'money')
# 2 9 核心
k, rk = M['core9'], R['core9_fixed_capital']
add('總 PnL', k['total_pnl'], rk['total_pnl'], 'money')
add('年化 Sharpe', k['sharpe'], rk['sharpe'])
add('Sortino', k['sortino'], rk['sortino'])
add('年化 PnL（日曆年 365.25）', k['ann_pnl_calendar'], rk['ann_pnl'], 'money',
    '原報告用日曆年；若用 n/252 年為 %.0f（差 2.6%%）' % k['ann_pnl_252'])
add('MDD（固定本金）', k['mdd'], rk['mdd'], 'money')
add('MDD 日期（固定本金）', k['mdd_date'], rk['mdd_date'], 'date')
add('Calmar（日曆年）', k['calmar_calendar'], rk['calmar'], note='若用 n/252 年為 %.4f' % k['calmar_252'])
add('日勝率', k['win_rate_daily'], rk['win_rate_daily'])
add('日盈虧比', k['pl_ratio_daily'], rk['pl_ratio_daily'])
add('交易筆數', k['n_trades'], rk['n_trades'], 'int')
add('逐筆勝率（含未平倉）', k['trade_win_rate'], R['trade_level']['win_rate_all'])
add('逐筆盈虧比（報酬率口徑）', k['trade_pl_ratio_ret'], R['trade_level']['pl_ratio_trade'],
    note='以金額算為 %.3f；原報告用報酬率平均' % k['trade_pl_ratio_pnl'])
# 3 檢定
add('T1 HAC t', mt['t1']['t'], rt['t01_hac']['t'])
add('T1 HAC p', mt['t1']['p'], rt['t01_hac']['p'])
add('T2 Lobato SR', mt['t2']['sr'], rt['t02_lobato']['sr'])
add('T2 Lobato 下限', mt['t2']['lo'], rt['t02_lobato']['lo'])
add('T2 Lobato 上限', mt['t2']['hi'], rt['t02_lobato']['hi'])
add('T3 PSR', mt['t3']['psr'], rt['t03_psr'])
add('T4 Bootstrap 2.5%', mt['t4']['lo'], rt['t04_bootstrap']['lo'])
add('T4 Bootstrap 97.5%', mt['t4']['hi'], rt['t04_bootstrap']['hi'])
for seg, v in mt['t6']['sub'].items():
    add(f'T6 子期 {seg} SR', v['sr'], rt['t06_subperiods'][seg]['sr'])
    add(f'T6 子期 {seg} PnL', v['pnl'], rt['t06_subperiods'][seg]['pnl'], 'money')
add('T7 Rolling252 平均 SR', mt['t7']['mean'], rt['t07_rolling252']['mean'])
add('T7 Rolling252 SR<0 比例', mt['t7']['pct_below0'], rt['t07_rolling252']['pct_below_0'])
add('T8 水下比例', mt['t8']['underwater'], rt['t08_drawdown']['underwater'])
add('T11 日版 Top5% 貢獻', mt['t11']['daily_top5_contrib'], rt['t11_concentration']['daily_top5'])
add('T11 偏度', mt['t11']['skew'], rt['t11_concentration']['skew'])
add('T11 峰度', mt['t11']['kurt'], rt['t11_concentration']['kurt'])
add('T11 逐筆前 5% 筆數', mt['t11']['trade_top5_n'], rt['t11_concentration']['trade_level']['n_top'], 'int')
add('T11 逐筆前 5% 占選股損益', mt['t11']['trade_top5_contrib'],
    rt['t11_concentration']['trade_level']['share_of_trade_pnl'])
add('T12 第三週三 PnL 占比', mt['t12']['settle_pct'], rt['t12_settlement']['settle_pct'])
add('T12 結算後首日占比', mt['t12']['post_pct'], rt['t12_settlement']['post_pct'])
add('T12 第三週三天數', mt['t12']['n_settle'], rt['t12_settlement']['n_settle'], 'int')
add('T12 換股執行日占比（加列，非核心）', mt['t12']['exec_day_pct'], rt['t12_settlement']['exec_day_pct'], 'ratio_strict',
    note='執行日由規則與逐筆檔推得（226 天），原報告用引擎日曆（225 天）；無引擎無法對齊')
add('T12 訊號日占比（加列，非核心）', mt['t12']['signal_day_pct'], rt['t12_settlement']['sig_day_pct'], 'ratio_strict',
    note='同上')
cm, rcm = mt['t13'], rt['t13_capm']
for lab, a, b in (('規格公式 vs TAIEX', 'spec_vs_taiex', 'vs_taiex_spec_formula'),
                  ('實際日報酬 vs TAIEX', 'actual_vs_taiex', 'vs_taiex_actual_daily_return'),
                  ('實際日報酬 vs 0050', 'actual_vs_0050', 'vs_0050_actual_daily_return')):
    add(f'T13 {lab} α 年化', cm[a]['alpha_ann'], rcm[b]['alpha_a'])
    add(f'T13 {lab} t(α)', cm[a]['t_alpha'], rcm[b]['t_alpha'])
    add(f'T13 {lab} β', cm[a]['beta'], rcm[b]['beta'])
    add(f'T13 {lab} R²', cm[a]['r2'], rcm[b]['r2'])
orig14 = {str(r['n_trials']): r['prob'] for r in rt['t14_dsr']['rows']}
orig14[str(rt['t14_dsr']['ours']['n_trials'])] = rt['t14_dsr']['ours']['prob']
for N in ('10', '50', '200', '1000', '4500', '4040'):
    add(f'T14 DSR N={N}', mt['t14']['dsr'][N]['prob'], orig14[N])
# 4 MC / 滾動
mc, rmc = M['monte_carlo'], R['monte_carlo']
for p in ('5', '50', '95'):
    add(f'MC 終值 P{p}', mc['final_p' + p], rmc['final'][p], 'mc')
    add(f'MC MDD P{p}', mc['mdd_p' + p], rmc['mdd'][p], 'mc')
for w in ('63', '252', '504'):
    add(f'滾動 {w} 日勝率', M['rolling'][w]['win_rate'], R['rolling'][w]['win_rate'])
    add(f'滾動 {w} 日中位 PnL', M['rolling'][w]['median_pnl'], R['rolling'][w]['median_pnl'], 'money')
    add(f'滾動 {w} 日 P5 PnL', M['rolling'][w]['p5_pnl'], R['rolling'][w]['p5_pnl'], 'money')
# 5 對照組
cb = R['control_benchmark']
for me, ok in (('strategy', 'strategy'), ('etf0050', 'b0050'), ('excess', 'excess')):
    add(f'對照 {me} SR', M['control'][me]['sr'], cb[ok]['sr'])
    add(f'對照 {me} HAC t', M['control'][me]['hac_t'], cb[ok]['hac_t'])
    add(f'對照 {me} DSR N=200', M['control'][me]['dsr_200'], cb[ok]['dsr200'])
    add(f'對照 {me} DSR N=4040', M['control'][me]['dsr_4040'], cb[ok]['dsr_ours'])
# 6 判定
mv = M['verdicts']
for i, row in enumerate(R['summary_rows'], 1):
    tag = '（沿用原值，未獨立重算）' if str(i) in mv['borrowed_not_recomputed'] else ''
    add(f'第 {i} 項判定符號{tag}', mv['symbols'][str(i)], row['verdict'][0], 'verdict')
for kk in ('passes', 'warns', 'triangs', 'crosses'):
    add(f'計數 {kk}', mv['counts'][kk], R['counts'][kk], 'int')
for kk in ('statistical', 'economic', 'overfit', 'stability'):
    add(f'verdict {kk}', mv[kk], R['verdicts'][kk], 'verdict')
add('敏感度：T13 改實際日報酬後 passes', mv['sensitivity_t13_actual_return']['counts']['passes'], 8, 'int',
    '原值 8 取自 回測標準.txt 第 8 節「存疑」文字')
add('敏感度：T13 改實際日報酬後 統計顯著', mv['sensitivity_t13_actual_return']['statistical'], '✓ 顯著', 'verdict',
    '原值取自 回測標準.txt 第 8 節')

core = [r for r in rows if '非核心' not in r['item']]
summary = {'n_items': len(rows), 'n_ok': sum(r['ok'] for r in rows),
           'n_core': len(core), 'n_core_ok': sum(r['ok'] for r in core),
           'not_ok': [r['item'] for r in rows if not r['ok']]}
json.dump({'summary': summary, 'rows': rows}, open(os.path.join(HERE, 'compare.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(summary)
for r in rows:
    if r['kind'] in ('date', 'verdict', 'int'):
        print(f"{'OK ' if r['ok'] else 'XX '}{r['item']}: {r['mine']} | {r['orig']}")
    else:
        print(f"{'OK ' if r['ok'] else 'XX '}{r['item']}: {r['mine']:.6g} | {r['orig']:.6g} | 差 {r['diff']:.3g} 相對 {r['rel_diff']:.2e}")
