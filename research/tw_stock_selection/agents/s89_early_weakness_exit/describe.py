#!/usr/bin/env python3
"""S89 全樣本描述（驗證期跑完之後才執行；會檢查 out/results.json 已存在）：現行標準路徑全部交易（已平倉 381＋未平倉 8），
依進場後第 3／5／10 個交易日累計報酬分五組 × 最後報酬；大贏家（已平倉報酬前 5%）／大輸家（後 5%）落在哪一組；
大贏家頭 5／10 天為負或曾跌超過 5%／8% 的筆數；13 種觸發條件的逐筆靜態對照（不考慮資金連動；毛報酬，不含成本）。
用法：python3 -I -B describe.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
assert (HERE / 'out' / 'results.json').exists(), '驗證期還沒跑完，不得先看全樣本逐筆描述'
sys.path.insert(0, str(HERE))
import s89lib as L
t0 = time.time()
P, U, RT = L.load()
S = L.signal(P, U, RT, 0)
nav, lg, st = L.sim(P, S, U, cap=1e6)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
TB = lastb('2022-01-01')
lg = lg.join(L.early_path(P, lg))
lg['closed'] = lg['why'] != 'open_end'
cl = lg[lg['closed']]
assert len(cl) == 381 and len(lg) == 389
n5 = int(round(len(cl) * 0.05))
lg['bigwin'] = False; lg['biglose'] = False
lg.loc[cl.index, 'bigwin'] = cl['ret'].rank(ascending=False, method='first') <= n5
lg.loc[cl.index, 'biglose'] = cl['ret'].rank(ascending=True, method='first') <= n5
lg['pnl_rank'] = lg['pnl'].where(lg['closed']).rank(ascending=False, method='first')
lg['period'] = np.where(lg['sig_date'] <= TB, 'train', 'valid')
cl = lg[lg['closed']]
tot_pnl = float(cl['pnl'].sum())
BINS = [-np.inf, -0.08, -0.03, 0.03, 0.08, np.inf]; LAB = ['≤−8%', '−8～−3%', '−3～+3%', '+3～+8%', '>+8%']
q = lambda y, p: float(y.quantile(p)) if len(y) else None
OUT = dict(n_closed=int(len(cl)), n_open=int((~lg['closed']).sum()), n5=n5, thr_bigwin=float(cl.loc[cl['bigwin'], 'ret'].min()), thr_biglose=float(cl.loc[cl['biglose'], 'ret'].max()),
           bigwin_pnl_share=float(cl.loc[cl['bigwin'], 'pnl'].sum() / tot_pnl), biglose_pnl_share=float(cl.loc[cl['biglose'], 'pnl'].sum() / tot_pnl), total_pnl=tot_pnl, groups={})
for n in (3, 5, 10):
    g = pd.cut(lg[f'r{n}'], BINS, labels=LAB, right=True)
    tb = {}
    for lab in LAB:
        y = lg[(g == lab) & lg['closed']]
        tb[lab] = dict(n=int(len(y)), mean=float(y['ret'].mean()), median=float(y['ret'].median()), p10=q(y['ret'], .1), p90=q(y['ret'], .9), win=float((y['ret'] > 0).mean()), pnl_sum=float(y['pnl'].sum()),
                       pnl_share=float(y['pnl'].sum() / tot_pnl), bigwin=int(y['bigwin'].sum()), biglose=int(y['biglose'].sum()), n_open=int(((g == lab) & ~lg['closed']).sum()))
    OUT['groups'][f'day{n}'] = tb
    OUT['groups'][f'day{n}_na'] = int(lg[f'r{n}'].isna().sum())


def early(y):
    o = dict(n=int(len(y)))
    for n in (3, 5, 10):
        o[f'r{n}_median'] = float(y[f'r{n}'].median()); o[f'r{n}_neg'] = int((y[f'r{n}'] < 0).sum())
        for k in (3, 5, 8, 10):
            o[f'r{n}_le_m{k}'] = int((y[f'r{n}'] <= -k / 100).sum())
        for k in (5, 8):
            o[f'min{n}_le_m{k}'] = int((y[f'min{n}'] <= -k / 100).sum())
    return o


OUT['early'] = dict(bigwin=early(cl[cl['bigwin']]), biglose=early(cl[cl['biglose']]), others=early(cl[~cl['bigwin'] & ~cl['biglose']]), all_closed=early(cl), top5_pnl=early(cl[cl['pnl_rank'] <= 5]))
cols = ['code', 'ent_date', 'exit_date', 'ret', 'pnl', 'pnl_rank', 'hold_days', 'r3', 'r5', 'r10', 'min5', 'min10']
fmt = lambda y: [{k: (str(v.date()) if hasattr(v, 'date') else (float(v) if isinstance(v, (float, np.floating)) else v)) for k, v in r.items()} for r in y[cols].to_dict('records')]
OUT['bigwin_list'] = fmt(cl[cl['bigwin']].sort_values('ret', ascending=False))
OUT['biglose_list'] = fmt(cl[cl['biglose']].sort_values('ret'))

# ---- 13 種觸發的逐筆靜態對照（含未平倉；每筆用該筆的實際買進成本當部位大小）
RULES = {c: v for c, v in L.CELLS.items() if not v['half']}
ST = {}
rows = []
for c, v in RULES.items():
    ev = []
    for r in lg.itertuples():
        o = L.stop_effect(P, S, int(r.j), int(r.ent_t), v['n'], v['x'], v['strict'], float(r.cost))
        if o is None:
            continue
        ev.append(dict(rule=c, code=str(r.code), ent_date=str(r.ent_date.date()), period=r.period, closed=bool(r.closed), ret=float(r.ret), pnl=float(r.pnl), bigwin=bool(r.bigwin), biglose=bool(r.biglose),
                       pnl_rank=(int(r.pnl_rank) if r.closed else None), **{k: o[k] for k in ('r_trig', 'g_stop', 'g_cur', 'd_stock', 'd_e50', 'd_total')}))
    E = pd.DataFrame(ev); rows.append(E)

    def sm(y):
        return dict(n=int(len(y)), d_stock=float(y['d_stock'].sum()), d_e50=float(y['d_e50'].sum()), d_total=float(y['d_total'].sum()),
                    mean_g_stop=float(y['g_stop'].mean()) if len(y) else None, mean_g_cur=float(y['g_cur'].mean()) if len(y) else None,
                    median_g_stop=float(y['g_stop'].median()) if len(y) else None, median_g_cur=float(y['g_cur'].median()) if len(y) else None,
                    frac_cur_better=float((y['g_cur'] > y['g_stop']).mean()) if len(y) else None)
    ST[c] = dict(all=sm(E), train=sm(E[E['period'] == 'train']), valid=sm(E[E['period'] == 'valid']), bigwin=sm(E[E['bigwin']]), biglose=sm(E[E['biglose']]),
                 others=sm(E[~E['bigwin'] & ~E['biglose']]), top5_pnl_hit=[f"{x.code}@{x.ent_date[:7]}" for x in E[E['pnl_rank'].fillna(99) <= 5].itertuples()],
                 bigwin_hit=[dict(code=x.code, ent_month=x.ent_date[:7], period=x.period, r_trig=x.r_trig, g_stop=x.g_stop, ret=x.ret, d_total=x.d_total) for x in E[E['bigwin']].itertuples()],
                 biglose_hit=[dict(code=x.code, ent_month=x.ent_date[:7], period=x.period, r_trig=x.r_trig, g_stop=x.g_stop, ret=x.ret, d_total=x.d_total) for x in E[E['biglose']].itertuples()])
OUT['static_rules'] = ST
OUT['static_note'] = '靜態對照＝每筆交易單獨看：觸發後次日開盤賣（毛報酬）對照照現行規則抱到出場（毛報酬），金額＝該筆買進成本 ×（差）＋賣得金額放 0050 到原出場日；不含多出來的成本、不考慮資金與名額連動，所以不等於淨值差。'
pd.concat(rows).to_csv(HERE / 'out' / 'static_rule_events.csv.gz', index=False)
keep = ['code', 'sig_date', 'ent_date', 'exit_date', 'why', 'ret', 'pnl', 'pnl_rank', 'hold_days', 'cost', 'r3', 'r5', 'r10', 'min3', 'min5', 'min10', 'bigwin', 'biglose', 'period', 'closed']
lg[keep].to_csv(HERE / 'out' / 'trades_early_path.csv.gz', index=False)
OUT['meta'] = dict(secs=round(time.time() - t0), peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
json.dump(OUT, open(HERE / 'out' / 'describe.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: OUT[k] for k in ('n_closed', 'n_open', 'n5', 'thr_bigwin', 'thr_biglose', 'bigwin_pnl_share', 'groups', 'early', 'meta')}, ensure_ascii=False))
for c in ST:
    print(c, {k: (ST[c][k]['n'], round(ST[c][k]['d_total'])) for k in ('all', 'train', 'valid', 'bigwin', 'biglose', 'others')}, ST[c]['top5_pnl_hit'])
