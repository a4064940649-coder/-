"""S66 主程式：依 PREREG 計算。python3 -I -B run.py"""
import json, sys, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s66lib as L
assert L.sha(L.HERE / 'SPEC.json') == (L.HERE / 'SPEC.sha256').read_text().strip()
assert L.sha(L.HERE / 'PREREG.json') == (L.HERE / 'PREREG.sha256').read_text().strip()
d = L.load_daily(); e = L.load_div()
cal = pd.DatetimeIndex(sorted(d.date.unique())); cal = cal[cal <= L.END]
navf = pd.read_csv(L.NAV_CSV, parse_dates=['date']).set_index('date')
STR = navf[L.STRAT].reindex(cal).ffill(); STR = STR[STR.first_valid_index():]
B50off = navf['0050'].reindex(cal).ffill().dropna()
TR, PX, META = {}, {}, {}
for c in L.MAIN + L.SUPP:
    tr, px, m = L.build_tr(d, e, c)
    TR[c] = tr.reindex(cal).ffill().dropna(); PX[c] = px.reindex(cal).ffill().dropna(); META[c] = m
B50 = TR['0050']
res = {'spec_sha256': (L.HERE / 'SPEC.sha256').read_text().strip(), 'prereg_sha256': (L.HERE / 'PREREG.sha256').read_text().strip(), 'end': str(L.END.date()), 'build_meta': META}

# ---------------------------------------------------------------- 驗收
def ext(path, col='adj', code=None):
    x = pd.read_csv(path, dtype={'code': str}, parse_dates=['date'])
    if code: x = x[x.code == code]
    return x.set_index('date')[col].sort_index()
EXT = {'0050': ext(L.FINLAB + 'etf_0050.csv'), '0056': ext(L.FINLAB + 'etf_0056.csv')}
for c in ['006208', '00878', '00919', '0050', '0056']:
    EXT[c + '_long'] = ext(L.EXTRAS + 'etf_long.csv.gz', code=c)
val = {}
for k, x in EXT.items():
    c = k.split('_')[0]; idx = TR[c].index.intersection(x.index)
    a, b = TR[c].reindex(idx), x.reindex(idx)
    val['自建 %s vs %s' % (c, 'etf_long' if k.endswith('_long') else 'finlab')] = dict(a=str(idx[0].date()), b=str(idx[-1].date()), cagr_own=L.cagr(a), cagr_ext=L.cagr(b), diff_pp=(L.cagr(a) - L.cagr(b)) * 100,
        max_abs_daily_ret_diff=float((a.pct_change() - b.pct_change()).abs().max()), pass_0p3=bool(abs(L.cagr(a) - L.cagr(b)) * 100 <= 0.3))
trm = pd.read_csv(L.EXTRAS + 'etf_tr_m.csv.gz', dtype={'code': str}); vm = {}
for c in L.MAIN + L.SUPP:
    x = trm[trm.code == c].set_index('ym').tr; x.index = pd.PeriodIndex(x.index, freq='M')
    o = TR[c].groupby(TR[c].index.to_period('M')).tail(1); o.index = o.index.to_period('M')
    j = pd.concat([o.pct_change(), x.pct_change()], axis=1, keys=['o', 'x']).dropna()
    j = j[j.index < pd.Period('2026-10', 'M')]
    if len(j): vm[c] = dict(n=len(j), med_abs_diff_pp=float((j.o - j.x).abs().median() * 100), max_abs_diff_pp=float((j.o - j.x).abs().max() * 100), cum_own=float((1 + j.o).prod() - 1), cum_ext=float((1 + j.x).prod() - 1))
val['自建 vs etf_tr_m 月報酬（2020-08 起，不含 2026-10 未滿月）'] = vm
i0 = B50off.index[0]
val['自建 0050 vs 正式淨值檔 0050 欄'] = dict(a=str(i0.date()), cagr_own=L.cagr(B50.loc[i0:]), cagr_official_col=L.cagr(B50off), diff_pp=(L.cagr(B50.loc[i0:]) - L.cagr(B50off)) * 100)
res['validation'] = val

# ---------------------------------------------------------------- 2015 以前接續
SPL = {}
for c, x in [('0050', EXT['0050']), ('0056', EXT['0056']), ('006208', EXT['006208_long'])]:
    t0 = TR[c].index[0]; pre = x[x.index <= t0]
    SPL[c] = pd.concat([pre.iloc[:-1] / pre.iloc[-1] * TR[c].iloc[0], TR[c]])
B50L = SPL['0050']

# ---------------------------------------------------------------- 各檔各期間
BASE = {'2019起': cal[cal < '2019-01-01'][-1], '2022起': cal[cal < '2022-01-01'][-1], '2024起': cal[cal < '2024-01-01'][-1]}
res['window_base'] = {k: str(v.date()) for k, v in BASE.items()}
S0 = STR.index[0]
def rel(c):
    f = TR[c].index[0]
    if f == cal[0]: return '2015 起（上市早於資料起點）'
    y = (L.END - f).days / 365.25
    return '樣本短、不可靠（上市 %.1f 年）' % y if y < 5 else '樣本中等（上市 %.1f 年）' % y
def trio(nav, a, b50=None):
    b50 = B50 if b50 is None else b50
    o = {'etf': L.stats(nav, b50, a, L.END), '0050': L.stats(b50, b50, a, L.END)}
    if a >= S0:
        o['策略'] = L.stats(STR, b50, a, L.END)
        o['etf_minus_策略_cagr_pp'] = (o['etf']['cagr'] - o['策略']['cagr']) * 100
        o['maxdd_etf_minus_策略_pp'] = (o['etf']['maxdd'] - o['策略']['maxdd']) * 100
    o['etf_minus_0050_cagr_pp'] = (o['etf']['cagr'] - o['0050']['cagr']) * 100
    return o
per = {}
for c in L.MAIN + L.SUPP:
    f = TR[c].index[0]; w = {}
    w['自上市起（官方日K＋twt49u）'] = trio(TR[c], f)
    w['自上市起（官方日K＋twt49u）']['不含息年化'] = L.cagr(PX[c].loc[f:L.END])
    if c in SPL:
        w['自上市起（含 2015 以前第三方還原價）'] = trio(SPL[c], SPL[c].index[0] if c != '006208' else SPL[c].index[0], B50L)
    a = max(f, S0); w['與策略重疊最長'] = trio(TR[c], a); w['與策略重疊最長']['不含息年化'] = L.cagr(PX[c].loc[a:L.END])
    for k, b in BASE.items():
        w[k] = trio(TR[c], b) if f <= b else '不適用（基底日尚未上市）'
        if f <= b: w[k]['不含息年化'] = L.cagr(PX[c].loc[b:L.END])
    yr = TR[c].groupby(TR[c].index.year).tail(1); yb = yr.pct_change().dropna()
    per[c] = {'名稱': d[d.stock_id == c].name.iloc[-1], '類別': L.CAT[c], '主表或補充': '主表' if c in L.MAIN else '補充', '首日': str(f.date()), '可靠度': rel(c),
              '含息': '可（官方收盤＋twt49u，除息 %d 次）' % META[c]['n_div'], 'windows': w,
              '逐年報酬（首年不滿年者不列）': {str(int(t.year)): float(v) for t, v in yb.items()}}
res['per_etf'] = per
# 策略本身
sw = {'全期（2015-07-01 起）': S0, **BASE}; res['策略'] = {}
for k, a in sw.items():
    o = {'vs 自建0050': L.stats(STR, B50, a, L.END), 'vs 正式淨值檔0050欄': L.stats(STR, B50off, a, L.END), '0050（自建）': L.stats(B50, B50, a, L.END)}
    res['策略'][k] = o
yr = STR.groupby(STR.index.year).tail(1); res['策略']['逐年報酬'] = {str(t.year): float(v) for t, v in yr.pct_change().dropna().items()}

# ---------------------------------------------------------------- 簡單組合
def verdict(cb, b0):
    better = cb['cagr'] > b0['cagr'] and cb['calmar'] > b0['calmar']
    worse = cb['cagr'] < b0['cagr'] and cb['calmar'] < b0['calmar']
    t = cb['t_nw6']
    if better: return '歷史上較好' + ('、統計上分得出來' if t == t and t >= 2 else '、統計上分不出來')
    if worse: return '較差' + ('（統計上分得出來）' if t == t and t <= -2 else '')
    return '分不出來（年化與 Calmar 一好一壞）'
comb = {}
others = [c for c in L.MAIN if c != '0050'] + ['策略']
for c in others:
    nav = STR if c == '策略' else TR[c]; f = nav.index[0]; comb[c] = {}
    for wn, w in [('70/30', 0.7), ('50/50', 0.5)]:
        comb[c][wn] = {}
        for k, a in {'重疊最長': f, **BASE}.items():
            if f > a: comb[c][wn][k] = '不適用'; continue
            cb = L.combo(B50, nav, w, a, L.END); s = L.stats(cb, B50, a, L.END); b0 = L.stats(B50, B50, a, L.END)
            comb[c][wn][k] = dict(a=s['a'], years=s['years'], cagr=s['cagr'], maxdd=s['maxdd'], calmar=s['calmar'], cagr_0050=b0['cagr'], maxdd_0050=b0['maxdd'], calmar_0050=b0['calmar'],
                                  cagr_diff_pp=(s['cagr'] - b0['cagr']) * 100, maxdd_diff_pp=(s['maxdd'] - b0['maxdd']) * 100, ex_ann_pp=s['ex_ann_pp'], t_nw6=s['t_nw6'], 判讀=verdict(s, b0))
res['combo_0050_plus'] = comb
res['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
json.dump(res, open(L.HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=str)

# ---------------------------------------------------------------- 印表
P = lambda x: '%6.1f' % (x * 100) if isinstance(x, float) and x == x else '   n/a'
print('== 驗收'); 
for k, v in val.items():
    if 'etf_tr_m' in k:
        for c, u in v.items(): print('  trm', c, u)
    else: print(' ', k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
print('== 策略'); 
for k in sw:
    s = res['策略'][k]['vs 自建0050']; s2 = res['策略'][k]['vs 正式淨值檔0050欄']
    print(' %-20s cagr %s mdd %s calmar %.2f | 0050 %s mdd %s | ex %.1f t %.2f | vs官方欄 ex %.1f t %.2f' % (k, P(s['cagr']), P(s['maxdd']), s['calmar'], P(s['cagr_0050']), P(s['maxdd_0050']), s['ex_ann_pp'], s['t_nw6'], s2['ex_ann_pp'], s2['t_nw6']))
for wk in ['自上市起（含 2015 以前第三方還原價）', '自上市起（官方日K＋twt49u）', '與策略重疊最長', '2019起', '2022起', '2024起']:
    print('==', wk)
    for c in L.MAIN + L.SUPP:
        o = per[c]['windows'].get(wk)
        if o is None or isinstance(o, str): continue
        s = o['etf']; st = o.get('策略')
        print(' %-7s %s y%5.2f cagr %s px %s mdd %s cal %5.2f vol %s | 0050 %s mdd %s | vs0050 geo %+6.1f ex %+6.1f t %5.2f | 策略 %s mdd %s diff %s' % (
            c, s['a'], s['years'], P(s['cagr']), P(o.get('不含息年化', float('nan'))), P(s['maxdd']), s['calmar'], P(s['vol']), P(s['cagr_0050']), P(s['maxdd_0050']), s['geo_diff_pp'], s['ex_ann_pp'], s['t_nw6'],
            P(st['cagr']) if st else '', P(st['maxdd']) if st else '', '%+6.1f' % o['etf_minus_策略_cagr_pp'] if st else ''))
print('== 組合')
for c in others:
    for wn in ['70/30', '50/50']:
        for k, v in comb[c][wn].items():
            if isinstance(v, str): continue
            print(' %-7s %s %-5s %s cagr %s mdd %s cal %.2f | 0050 %s %s %.2f | d %+5.1f ddd %+5.1f t %5.2f %s' % (c, wn, k, v['a'], P(v['cagr']), P(v['maxdd']), v['calmar'], P(v['cagr_0050']), P(v['maxdd_0050']), v['calmar_0050'], v['cagr_diff_pp'], v['maxdd_diff_pp'], v['t_nw6'], v['判讀']))
print('== 逐年')
ys = [str(y) for y in range(2016, 2027)]
print('        ' + ' '.join('%6s' % y for y in ys))
for c in L.MAIN: print(' %-7s' % c + ' '.join(P(per[c]['逐年報酬（首年不滿年者不列）'].get(y, float('nan'))) for y in ys))
print(' 策略   ' + ' '.join(P(res['策略']['逐年報酬'].get(y, float('nan'))) for y in ys))
print('peak MB', res['peak_rss_mb'])
