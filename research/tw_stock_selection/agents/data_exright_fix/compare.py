#!/usr/bin/env python3
"""第 4 步：合併 classify.json、scratchpad/exr/res_*.json、list_2026_11.json → results.json（本資料夾）。python3 -I -B compare.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from exr_common import TMP

pp = lambda x: round(float(x) * 100, 2)
cl = json.load(open(HERE / 'classify.json'))
R = {m: json.load(open(TMP / f'res_{m}.json')) for m in ('orig', 'twt', 'main', 'close')}
D = {m: json.load(open(TMP / f'res_{m}_sigorig.json')) for m in ('twt', 'main', 'close') if (TMP / f'res_{m}_sigorig.json').exists()}
L26 = json.load(open(TMP / 'list_2026_11.json'))
o = R['orig']
NAMES = {'orig': '原樣（下界版：維持 0）', 'twt': '只修除權息（TWT49U 參考價）', 'main': '建議修法（TWT49U 參考價＋前一日無成交改收盤報酬）',
         'close': '收盤報酬版（所有 0 格改收盤÷前收，除權息當下跌）'}


def list_diff(a, b):
    months = sorted(set(a) | set(b)); out = []
    for k in months:
        x, y = set(a.get(k, [])), set(b.get(k, []))
        if x != y:
            out.append(dict(month=k, removed=sorted(x - y), added=sorted(y - x)))
    return out


def trade_ret(tl, code, month):
    for t in tl:
        if t['code'] == code and t['ent_date'][:7] == month:
            return round(t['ret'] * 100, 1)
    return None


res = dict(baseline_reported=dict(cagr_since2019=48.13, cagr_full=34.08, maxdd=-33.40, n_trades_closed=379, paths_median_since2019=46.23, paths_median_maxdd=-34.01),
           classification=cl, variants={})
for m, r in R.items():
    s = r['std']; ps = r['paths_summary']
    dif = np.array([r['paths'][k] - o['paths'][k] for k in o['paths']])
    ld = list_diff(o['lists_f1'], r['lists_f1'])
    for x in ld:
        x['removed_ret_orig_pct'] = {c: trade_ret(o['trades_list'], c, x['month'][:4] + '-' + x['month'][5:7]) for c in x['removed']}
        x['added_ret_new_pct'] = {c: trade_ret(r['trades_list'], c, x['month'][:4] + '-' + x['month'][5:7]) for c in x['added']}
    res['variants'][m] = dict(
        name=NAMES[m], n_chg_cells_changed=r['n_chg_cells'], n_ref_fallback_rights_issue=r.get('n_ref_fallback_rights_issue'),
        std=dict(cagr_since2019=pp(s['cagr_since2019']), cagr_full=pp(s['cagr_full']), cagr_since2024=pp(s['cagr_since2024']), maxdd=pp(s['maxdd']),
                 dd_peak=s['dd_peak'], dd_trough=s['dd_trough'], n_trades_closed=s['n_trades_closed'], n_trades_all=s['n_trades_all'],
                 win_rate=pp(s['trades']['win_rate']), avg_win=pp(s['trades']['avg_win']), avg_loss=pp(-s['trades']['avg_loss']), open_end=s['open_end'],
                 yearly={y: pp(v) for y, v in s['yearly'].items()}),
        diff_vs_orig_pp=dict(cagr_since2019=round(pp(s['cagr_since2019']) - pp(o['std']['cagr_since2019']), 2), cagr_full=round(pp(s['cagr_full']) - pp(o['std']['cagr_full']), 2),
                             maxdd=round(pp(s['maxdd']) - pp(o['std']['maxdd']), 2),
                             yearly={y: round(pp(v) - pp(o['std']['yearly'][y]), 2) for y, v in s['yearly'].items()}),
        paths=dict(cagr19_median=pp(ps['cagr19_median']), cagr19_min=pp(ps['cagr19_min']), cagr19_max=pp(ps['cagr19_max']), maxdd_median=pp(ps['maxdd_median']), maxdd_worst=pp(ps['maxdd_worst']),
                   paired_diff_since2019_pp=dict(median=round(float(np.median(dif)) * 100, 2), min=round(float(dif.min()) * 100, 2), max=round(float(dif.max()) * 100, 2), n_pos=int((dif > 0).sum()), n=int(len(dif)))),
        f1_lists=dict(n_signal_months=len(r['lists_f1']), n_picks_orig=sum(len(v) for v in o['lists_f1'].values()), n_months_changed=len(ld),
                      n_removed=sum(len(x['removed']) for x in ld), n_added=sum(len(x['added']) for x in ld), changes=ld),
        returns_only_with_orig_signals=(dict(cagr_since2019=pp(D[m]['cagr_since2019']), cagr_full=pp(D[m]['cagr_full']), maxdd=pp(D[m]['maxdd']), n_trades_closed=D[m]['n_trades_closed'])
                                        if m in D else None))
# 報酬面：同一批交易（原訊號）逐筆報酬差
if 'main' in D:
    a = pd.DataFrame(o['trades_list']); b = pd.DataFrame(D['main']['trades_list'])
    mm = a.merge(b, on=['code', 'ent_date'], suffixes=('_o', '_n')); mm['d'] = mm['ret_n'] - mm['ret_o']
    res['trade_level_main_same_signals'] = dict(n_trades=int(len(mm)), n_changed=int((mm['d'].abs() > 1e-9).sum()), mean_diff_pp=round(float(mm['d'].mean()) * 100, 3),
                                                worst=mm.sort_values('d').head(5)[['code', 'ent_date', 'ret_o', 'ret_n']].round(4).to_dict(orient='records'),
                                                best=mm.sort_values('d').tail(5)[['code', 'ent_date', 'ret_o', 'ret_n']].round(4).to_dict(orient='records'))
res['list_2026_11'] = {m: dict(n_ma=v['n_ma'], top20=v['top20'], f1=v['f1'], top20_same_as_orig=v['top20'] == L26['orig']['top20'], f1_same_as_orig=v['f1'] == L26['orig']['f1'],
                               e0_ranks=v['e0'], rows=v['rows']) for m, v in L26.items()}
res['list_2026_11_unfixable_zero_cells_after_twt_mirror_end'] = L26['orig']['log']['zero_cells_twse_after_twt_last']
m = res['variants']['main']
res['verdict'] = dict(rule='影響 ≤ 1 個百分點且名單幾乎不變 → 不影響結論',
                      main_abs_diff_since2019_pp=abs(m['diff_vs_orig_pp']['cagr_since2019']), main_abs_diff_full_pp=abs(m['diff_vs_orig_pp']['cagr_full']),
                      main_paths_median_diff_pp=m['paths']['paired_diff_since2019_pp']['median'],
                      list_2026_11_unchanged=res['list_2026_11']['main']['top20_same_as_orig'] and res['list_2026_11']['main']['f1_same_as_orig'])
res['verdict']['result'] = '不影響結論' if (res['verdict']['main_abs_diff_since2019_pp'] <= 1 and res['verdict']['main_abs_diff_full_pp'] <= 1 and res['verdict']['list_2026_11_unchanged']) else '有影響'
json.dump(res, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=str)
for k, v in res['variants'].items():
    print(k, v['std']['cagr_since2019'], v['std']['cagr_full'], v['std']['maxdd'], v['std']['n_trades_closed'], v['paths']['cagr19_median'], v['paths']['maxdd_median'],
          v['paths']['paired_diff_since2019_pp'], v['f1_lists']['n_months_changed'], v['f1_lists']['n_removed'], v['f1_lists']['n_added'], v['returns_only_with_orig_signals'])
print(res['verdict'])
