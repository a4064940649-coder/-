#!/usr/bin/env python3
"""S96 附帶描述（不登記、不判定）：S75 連續帳戶 2008–2026（v2、V_BRIDGE 632 筆）裡，訊號日在 2012 年後、有官方估值資料的已平倉交易，
大贏家（632 筆已平倉中報酬率前 5%）進場訊號日的本益比／淨值比分佈，對照全部交易。
同日百分位＝在當日全部上市四碼股票（有值者）裡的百分位（不是策略股池）。用法：python3 -I -B desc632.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent / 'raw' / 'tsd_extra'
sys.path.insert(0, str(RAW))
from tsd_extra_load import load as tload
t = pd.read_csv(HERE.parent / 's75_continuous_2008_2026' / 'v2_2026-11-21' / 'trades_v2.csv.gz', dtype={'code': str})
t = t[t['version'] == 'V_BRIDGE'].copy(); n_all = len(t)
c = t[t['why'] != 'open_end'].copy(); n5 = int(round(len(c) * 0.05))
c['bigwin'] = c['ret'].rank(ascending=False, method='first') <= n5; c['biglose'] = c['ret'].rank(ascending=True, method='first') <= n5
days = sorted(c.loc[c['sig_date'] >= '2012-01-02', 'sig_date'].unique())
rows = []
for name in ('per_twse_early', 'per_twse'):
    df = tload(name, columns=['date', 'code', 'per', 'pbr', 'yield_pct']); df['date'] = df['date'].astype(str)
    df = df[df['date'].isin(days) & df['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
    for k in ('per', 'pbr', 'yield_pct'):
        df[k] = df[k].astype('float64').round(2)
    df.loc[~(df['per'] > 0), 'per'] = np.nan
    df['p_pbr'] = df.groupby('date')['pbr'].rank(pct=True); df['p_per'] = df.groupby('date')['per'].rank(pct=True)
    rows.append(df)
v = pd.concat(rows, ignore_index=True).rename(columns={'date': 'sig_date'})
x = c[c['sig_date'] >= '2012-01-02'].merge(v, on=['sig_date', 'code'], how='left')
x['has'] = x['pbr'].notna()
cov = dict(n_632=n_all, n_closed=int(len(c)), n5=n5, thr_bigwin=float(c.loc[c['bigwin'], 'ret'].min()), n_closed_sig_2012on=int(len(x)), with_val_row=int(x['has'].sum()),
           bigwin_total=int(c['bigwin'].sum()), bigwin_2012on=int(x['bigwin'].sum()), bigwin_2012on_with_val=int((x['bigwin'] & x['has']).sum()), days_without_file=[d for d in days if d not in set(v['sig_date'])])
x = x[x['has']].copy()
x['g_per'] = np.where(x['per'].isna(), '0虧損或無', np.where(x['per'] < 15, '1<15', np.where(x['per'] <= 30, '2 15~30', '3>30')))
x['g_pbr'] = pd.cut(x['pbr'], [-np.inf, 1, 2, 4, np.inf], labels=['1 ≤1', '2 1~2', '3 2~4', '4 >4']).astype(str)
x['g_ppbr'] = pd.cut(x['p_pbr'], [-0.01, 1 / 3, 2 / 3, 0.8, 1.01], labels=['1全上市最低1/3', '2中1/3', '3 67~80%', '4最高20%']).astype(str)
x['g_pper'] = np.where(x['per'].isna(), '0虧損或無', pd.cut(x['p_per'], [-0.01, 1 / 3, 2 / 3, 0.8, 1.01], labels=['1全上市最低1/3', '2中1/3', '3 67~80%', '4最高20%']).astype(str))


def grp(lab):
    o = {}
    for g, y in x.groupby(lab):
        o[str(g)] = dict(n=int(len(y)), share_all=float(len(y) / len(x)), win=float((y['ret'] > 0).mean()), mean=float(y['ret'].mean()), median=float(y['ret'].median()),
                         bigwin=int(y['bigwin'].sum()), share_bigwin=float(y['bigwin'].sum() / max(1, x['bigwin'].sum())), biglose=int(y['biglose'].sum()))
    return o


q = lambda s: {f'p{int(p*100)}': float(s.quantile(p)) for p in (0.25, 0.5, 0.75)}
bw = x[x['bigwin']]
res = dict(note='不登記、不判定；樣本含現行策略訓練／驗證期，僅描述', coverage=cov, groups={k: grp(k) for k in ('g_per', 'g_pbr', 'g_ppbr', 'g_pper')},
           quantiles=dict(all=dict(per=q(x['per'].dropna()), pbr=q(x['pbr']), p_pbr=q(x['p_pbr']), yld=q(x['yield_pct'])), bigwin=dict(per=q(bw['per'].dropna()), pbr=q(bw['pbr']), p_pbr=q(bw['p_pbr']), yld=q(bw['yield_pct'])),
                          biglose=dict(per=q(x[x['biglose']]['per'].dropna()), pbr=q(x[x['biglose']]['pbr']), p_pbr=q(x[x['biglose']]['p_pbr']))),
           bigwin_list=[dict(code=r.code, sig_date=r.sig_date, ret=round(float(r.ret), 4), per=(None if np.isnan(r.per) else r.per), pbr=r.pbr, yld=r.yield_pct, p_pbr=round(r.p_pbr, 3),
                             p_per=(None if np.isnan(r.p_per) else round(r.p_per, 3))) for r in bw.sort_values('ret', ascending=False).itertuples()])
(HERE / 'desc632.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(cov)
for k, g in res['groups'].items():
    print(k, {a: (b['n'], round(b['share_all'], 2), round(b['win'], 2), round(b['mean'], 3), round(b['median'], 3), b['bigwin'], round(b['share_bigwin'], 2), b['biglose']) for a, b in g.items()})
print(res['quantiles'])
for r in res['bigwin_list']:
    print(r)
