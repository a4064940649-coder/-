#!/usr/bin/env python3
"""S86 主程式：散戶稅後費後摩擦逐項與合計。用法：python3 -I -B run.py → results.json、yearly_dividends.csv、payments_std.csv.gz。
所有稅率、費率、門檻都是「假設情境」參數（見 ASSUME），不是法規事實；讀者需自行查證現行規定。"""
import json, time, resource, sys, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s86lib as L
from s86lib import s10lib, stlib, COSTS
HERE = Path(__file__).resolve().parent
t00 = time.time()
ASSUME = dict(div_tax_scenarios=[-0.035, 0.0, 0.05, 0.12, 0.20, 0.28], nhi_rate=0.0211, nhi_thr=20000.0, delay_days=20,
              note='稅率、補充保費費率與門檻皆為假設情境；−3.5% 代表「併入申報、適用 5% 級距、可抵減 8.5% 而退稅」這種假設的有效稅率，未查證')
SHIFTS = (0, 1, 2, 3)
P, U, SG = L.setup(SHIFTS)
DY, DYM, NQ, dy50, e50ev, dinfo = L.div_arrays(P)
EXPY = L.expy(P, DY)
D0, DE = P.IDX[P.t0], P.IDX[-1]
D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
yrs = P.IDX.year.to_numpy()
c19 = lambda n: float(stlib.cagr(n, D19, DE)); cfu = lambda n: float(stlib.cagr(n, n.index[0], DE)); mdd = lambda n: float(stlib.maxdd(n, n.index[0], DE))
BASE = dict(sat_w=0.7, sizing='W10', lock=False)


def sim(s=0, cap=1e6, cost='strict', **kw):
    c = COSTS[cost] if isinstance(cost, str) else cost
    return L.simulate_s86(P, SG[s], U, cost=c, capital=cap, DY=kw.pop('DYx', DY), dy50=dy50, **BASE, **kw)


def m(n):
    return dict(c19=c19(n), full=cfu(n), maxdd=mdd(n))


R = dict(assume=ASSUME, div_info=dinfo, window=dict(start=str(D0.date()), since2019_from=str(D19.date()), end=str(DE.date())))
nav0, lg0, st0 = sim(EXPY=EXPY)
b0, _, bpay = L.bench50(P, dy50)
R['base'] = dict(strategy=m(nav0), e50=m(b0), n_closed=int((lg0['why'] != 'open_end').sum()))
assert abs(R['base']['strategy']['c19'] - L.ACC['since2019']) < 1e-12 and R['base']['n_closed'] == 381
print('base', R['base'], flush=True)

# ---------- 1. 各年收到的現金股利（占前一日淨值，逐日加總）
dv = pd.DataFrame(st0['_dv'], index=P.IDX)
_, _, stM = sim(DYx=DY + DYM)                                   # 上限：權息事件的「權值＋息值」全當現金
dvM = pd.DataFrame(stM['_dv'], index=P.IDX)
rows = []
for y in sorted(set(yrs[P.t0 + 1:])):
    k = (yrs == y) & (np.arange(P.T) > P.t0)
    d = dv[k]; dm_ = dvM[k]
    nz = d['nav_prev'] > 0
    rows.append(dict(year=int(y),
                     strat_div_pct_nav=float(((d['stk_gross'] + d['e50_gross']) / d['nav_prev'])[nz].sum()),
                     strat_stock_part=float((d['stk_gross'] / d['nav_prev'])[nz].sum()), strat_0050_part=float((d['e50_gross'] / d['nav_prev'])[nz].sum()),
                     strat_div_pct_nav_upper_mixed=float(((dm_['stk_gross'] + dm_['e50_gross']) / dm_['nav_prev'])[nz].sum()),
                     e50_only_div_pct=float(dy50[k].sum()),
                     stock_sleeve_yield=float((d['stk_gross'] / d['stk_val'].where(d['stk_val'] > 0)).sum()),
                     stock_sleeve_expected_timeprop=float((d['stk_exp'] / d['stk_val'].where(d['stk_val'] > 0)).sum()),
                     avg_stock_weight=float((d['stk_val'] / d['nav_prev'])[nz].mean()), avg_0050_weight=float((d['v50_val'] / d['nav_prev'])[nz].mean()),
                     ntd_stock=float(d['stk_gross'].sum()), ntd_0050=float(d['e50_gross'].sum())))
Y = pd.DataFrame(rows); Y.to_csv(HERE / 'yearly_dividends.csv', index=False)
f19 = Y[Y['year'] >= 2019]
R['dividends'] = dict(yearly=rows,
                      avg_2019on={c: float(f19[c].mean()) for c in Y.columns if c not in ('year', 'ntd_stock', 'ntd_0050')},
                      avg_2016on={c: float(Y[Y['year'] >= 2016][c].mean()) for c in Y.columns if c not in ('year', 'ntd_stock', 'ntd_0050')})
pay = pd.DataFrame(st0['_pay'], columns=['t', 'j', 'gross', 'nav_prev'])
pay['date'] = P.IDX[pay['t'].to_numpy()]; pay['code'] = np.where(pay['j'] < 0, '0050', P.COLS[np.maximum(pay['j'].to_numpy(), 0)]); pay['frac'] = pay['gross'] / pay['nav_prev']
pay[['date', 'code', 'gross', 'nav_prev', 'frac']].to_csv(HERE / 'payments_std.csv.gz', index=False)
print('div', R['dividends']['avg_2019on'], flush=True)

# ---------- 7. 是否常在除息前賣、除息後買
x = lg0.copy(); cap_ = dict(captured=0, sold_within20_before_ex=0, bought_within20_after_ex=0, stock_only_captured=0, mixed_captured=0)
capt_y = exp_y = 0.0
for r in x.itertuples():
    j, e, xt = int(r.j), int(r.ent_t), int(r.exit_t)
    hi = xt if r.why != 'open_end' else P.T - 1
    cap_['captured'] += int((DY[e + 1:hi + 1, j] > 0).sum())
    cap_['mixed_captured'] += int((DYM[e + 1:hi + 1, j] > 0).sum()); cap_['stock_only_captured'] += int(NQ[e + 1:hi + 1, j].sum())
    if r.why != 'open_end':
        cap_['sold_within20_before_ex'] += int((DY[xt + 1:xt + 21, j] > 0).sum())
    cap_['bought_within20_after_ex'] += int((DY[max(e - 19, 0):e + 1, j] > 0).sum())
ks = np.arange(P.T) > P.t0
tot_g = float(dv['stk_gross'][ks].sum()); tot_e = float(dv['stk_exp'][ks].sum())
R['q7'] = dict(event_counts=cap_, n_trades=int(len(x)), avg_hold_days=float(x['hold_days'].mean()),
               stock_div_ntd_actual=tot_g, stock_div_ntd_expected_timeprop=tot_e, capture_ratio_value_weighted=tot_g / tot_e,
               sleeve_yield_avg_2019on=float(f19['stock_sleeve_yield'].mean()), sleeve_expected_avg_2019on=float(f19['stock_sleeve_expected_timeprop'].mean()),
               e50_yield_avg_2019on=float(f19['e50_only_div_pct'].mean()),
               note='expected＝每天持股市值 × 該股當年現金股利率合計 ÷ 當年交易日數；ratio<1 表示持有期避開了除息日')
# 持股所屬股票的「整年現金殖利率」（市值加權）對照 0050
R['q7']['held_stocks_fullyear_yield_2019on'] = float((f19['stock_sleeve_expected_timeprop'] / 1.0).mean())
print('q7', R['q7'], flush=True)

# ---------- 1b. 股利稅情境（標準路徑＋4 個位移）
tax = {}
for tau in ASSUME['div_tax_scenarios']:
    b, _, _ = L.bench50(P, dy50, div_tax=tau)
    row = dict(e50=m(b), strat={})
    for s in SHIFTS:
        n, _, _ = sim(s, div_tax=tau); row['strat'][s] = m(n)
    tax[str(tau)] = row
n, _, _ = sim(0, div_tax=0.28, DYx=DY + DYM); tax['0.28_upper_mixed'] = dict(strat={0: m(n)})
R['tax'] = tax
base_s = {s: tax['0.0']['strat'][s] for s in SHIFTS}
print('tax', {k: (round((v['strat'][0]['c19'] - base_s[0]['c19']) * 100, 3), round((v['e50']['c19'] - R['base']['e50']['c19']) * 100, 3) if 'e50' in v else None) for k, v in tax.items()}, flush=True)

# ---------- 2. 補充保費：靜態（帳戶固定在 X 元）＋路徑（帳戶從 X 起跑會長大）
nhi = dict(static={}, path={})
pay['year'] = pay['date'].dt.year
bp = pd.DataFrame(bpay, columns=['t', 'j', 'gross', 'nav_prev']); bp['frac'] = bp['gross'] / bp['nav_prev']; bp['year'] = P.IDX[bp['t'].to_numpy()].year
ny19 = len(set(pay.loc[pay['year'] >= 2019, 'year']))
for X in (1e6, 3e6, 1e7):
    for thr in (20000.0, 0.0):
        o = {}
        for nm, q in (('strategy', pay), ('e50', bp)):
            q2 = q[q['year'] >= 2019]; hit = q2['frac'] * X >= thr
            o[nm] = dict(prem_pct_per_year=float((q2['frac'][hit] * ASSUME['nhi_rate']).sum() / ny19), prem_ntd_per_year=float((q2['frac'][hit] * ASSUME['nhi_rate']).sum() / ny19 * X),
                         n_pay_per_year=float(len(q2) / ny19), n_hit_per_year=float(hit.sum() / ny19), div_ntd_per_year=float(q2['frac'].sum() / ny19 * X),
                         max_single_ntd=float(q2['frac'].max() * X))
        nhi['static'][f'{int(X/1e4)}萬|門檻{int(thr)}'] = o
for X in (1e6, 3e6, 1e7):
    n1, _, s1 = sim(0, cap=X); n2, _, s2 = sim(0, cap=X, nhi_rate=ASSUME['nhi_rate'], nhi_thr=ASSUME['nhi_thr'])
    b1, _, _ = L.bench50(P, dy50, cap=X); b2, t2, _ = L.bench50(P, dy50, cap=X, nhi_rate=ASSUME['nhi_rate'], nhi_thr=ASSUME['nhi_thr'])
    nhi['path'][f'{int(X/1e4)}萬'] = dict(strat_base=m(n1), strat_nhi=m(n2), e50_base=m(b1), e50_nhi=m(b2), strat_nhi_ntd=float(s2['_dv']['nhi'].sum()), e50_nhi_ntd=float(t2['nhi']))
R['nhi'] = nhi
print('nhi', {k: (round(v['strategy']['prem_pct_per_year'] * 100, 4), round(v['e50']['prem_pct_per_year'] * 100, 4)) for k, v in nhi['static'].items()}, flush=True)

# ---------- 3. 股利入帳時間差
dl = {}
for d in (10, 20, 40):
    b, _, _ = L.bench50(P, dy50, delay=d)
    dl[d] = dict(e50=m(b), strat={s: m(sim(s, delay=d)[0]) for s in SHIFTS})
R['delay'] = dl
print('delay', {d: (round((v['strat'][0]['c19'] - base_s[0]['c19']) * 100, 3), round((v['e50']['c19'] - R['base']['e50']['c19']) * 100, 3)) for d, v in dl.items()}, flush=True)

# ---------- 4. 手續費折扣與低消
fee = dict(path={}, first_year={})
for cap, disc, mf in itertools.product((3e5, 5e5, 1e6), (0.28, 0.6, 1.0), (20.0, 0.0)):
    n, _, s_ = sim(0, cap=cap, fee_rate=L.FEE_FULL * disc, minfee_v=mf)
    fee['path'][f'{int(cap/1e4)}萬|{disc}|低消{int(mf)}'] = dict(**m(n), fee_ntd=float(s_['fee']), e50_cost_ntd=float(s_['e50_cost']))
# 第一年（帳戶還在原水位）：2018～2025 每年最後交易日全現金新起跑，只看之後一年；零股全期可買（現行制度）
ye = [int(np.flatnonzero(yrs == y)[-1]) for y in range(2018, 2026)]
ends = ye[1:] + [P.T - 1]
for cap in (3e5, 5e5, 1e6):
    for disc, mf in itertools.product((0.28, 0.6, 1.0), (20.0, 0.0)):
        v = []
        for (a, b_), s in itertools.product(zip(ye, ends), SHIFTS):
            n, _, s_ = L.simulate_s86(P, SG[s], U, cost=COSTS['strict'], capital=cap, start_t=a, odd_from='all', fee_rate=L.FEE_FULL * disc, minfee_v=mf, **BASE)
            v.append(float(n.iloc[b_ - a] / cap - 1))
        fee['first_year'][f'{int(cap/1e4)}萬|{disc}|低消{int(mf)}'] = dict(mean_ret=float(np.mean(v)), n=len(v), rets=v)
R['fee'] = fee
print('fee first-year', {k: round(v['mean_ret'] * 100, 2) for k, v in fee['first_year'].items()}, flush=True)

# ---------- 5. 零股價差：幾乎全用零股（每邊再加 0.3%／0.6%，全部成交都加）
odd = {}
for cap in (3e5, 5e5, 1e6):
    for nm, c in (('全部零股+0.3%', dict(cmult=1.0, xslip=0.006, oddx=0.0)), ('全部零股+0.6%', dict(cmult=1.0, xslip=0.009, oddx=0.0)),
                  ('只零股部分+0.3%', dict(cmult=1.0, xslip=0.003, oddx=0.003)), ('基準', COSTS['strict'])):
        odd[f'{int(cap/1e4)}萬|{nm}'] = {s: m(sim(s, cap=cap, cost=c, odd_from='all')[0]) for s in SHIFTS}
R['oddlot'] = odd
print('odd', {k: round(v[0]['c19'] * 100, 2) for k, v in odd.items()}, flush=True)

# ---------- 6. 合計：三種散戶情境
SC = {'小資_50萬_稅5%_6折': dict(cap=5e5, tau=0.05, disc=0.6), '一般_100萬_稅12%_2.8折': dict(cap=1e6, tau=0.12, disc=0.28), '高所得_500萬_稅28%_2.8折': dict(cap=5e6, tau=0.28, disc=0.28)}
comb = {}
for nm, sc in SC.items():
    fr = dict(div_tax=sc['tau'], nhi_rate=ASSUME['nhi_rate'], nhi_thr=ASSUME['nhi_thr'], delay=ASSUME['delay_days'], fee_rate=L.FEE_FULL * sc['disc'])
    o = dict(before={}, after={}, after_upper_mixed={}, after_plus_oddlot={})
    bb, _, _ = L.bench50(P, dy50, cap=sc['cap']); ba, ta, _ = L.bench50(P, dy50, cap=sc['cap'], **fr)
    o['e50_before'] = m(bb); o['e50_after'] = m(ba); o['e50_after_ntd'] = ta
    for s in SHIFTS:
        o['before'][s] = m(sim(s, cap=sc['cap'])[0])
        n, _, s_ = sim(s, cap=sc['cap'], **fr); o['after'][s] = m(n)
        if s == 0:
            o['after_ntd'] = dict(tax=float(s_['_dv']['tax'].sum()), nhi=float(s_['_dv']['nhi'].sum()), gross_div=float(s_['_dv']['stk_gross'].sum() + s_['_dv']['e50_gross'].sum()), final_nav=float(n.iloc[-1]))
        o['after_upper_mixed'][s] = m(sim(s, cap=sc['cap'], DYx=DY + DYM, **fr)[0])
        o['after_plus_oddlot'][s] = m(sim(s, cap=sc['cap'], cost=dict(cmult=1.0, xslip=0.003, oddx=0.003), **fr)[0])
    # 逐項拆解（標準路徑，各自單獨加一項）
    b0_ = o['before'][0]
    o['steps'] = {k: m(sim(0, cap=sc['cap'], **kw)[0]) for k, kw in (('只加股利稅', dict(div_tax=sc['tau'])), ('只加補充保費', dict(nhi_rate=ASSUME['nhi_rate'], nhi_thr=ASSUME['nhi_thr'])),
                                                                    ('只加入帳延後', dict(delay=ASSUME['delay_days'])), ('只改手續費折扣', dict(fee_rate=L.FEE_FULL * sc['disc'])))}
    o['e50_steps'] = {k: m(L.bench50(P, dy50, cap=sc['cap'], **kw)[0]) for k, kw in (('只加股利稅', dict(div_tax=sc['tau'])), ('只加補充保費', dict(nhi_rate=ASSUME['nhi_rate'], nhi_thr=ASSUME['nhi_thr'])),
                                                                               ('只加入帳延後', dict(delay=ASSUME['delay_days'])))}
    comb[nm] = o
    a0, e0, e1 = o['after'][0], o['e50_before'], o['e50_after']
    print(nm, '策略 2019起 %.2f→%.2f 全期 %.2f→%.2f｜0050 %.2f→%.2f 全期 %.2f→%.2f｜領先 %.2f→%.2f' % (
        b0_['c19'] * 100, a0['c19'] * 100, b0_['full'] * 100, a0['full'] * 100, e0['c19'] * 100, e1['c19'] * 100, e0['full'] * 100, e1['full'] * 100,
        (b0_['c19'] - e0['c19']) * 100, (a0['c19'] - e1['c19']) * 100), flush=True)
R['combined'] = comb
R['secs'] = round(time.time() - t00); R['peak_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
json.dump(R, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=str)
print('done', R['secs'], 's', R['peak_mb'], 'MB')
