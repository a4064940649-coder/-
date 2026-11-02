#!/usr/bin/env python3
"""第 5 步（事件層級）：樣本外（公告季 2019Q2～2026Q2）機制檢定 P1～P5、P7 前置、分市場、分年、分產業、去掉最大貢獻股。
定義與 s2_explore.py 相同（凍結後不改 s2）；需 PREREG。輸出 results.json 的 EVENT_OOS。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pcore import DATA, HERE, load_panel, e50_on, tstat, sha

R = json.load(open(HERE / 'results.json'))
assert 'PREREG' in R and R['PREREG']['hashes']['data/events.parquet'] == sha(DATA / 'events.parquet')
E = pd.read_parquet(DATA / 'events.parquet')
paths = np.load(DATA / 'event_paths.npz'); PATH, EWP = paths['PATH'], paths['EWP']
m = (E['k'] >= 2019 * 4 + 1) & E['SUEP'].notna()
E = E[m].copy(); PATH = PATH[m.to_numpy()]; EWP = EWP[m.to_numpy()]
P = load_panel(); IDX = pd.DatetimeIndex(P['IDX']); e50 = e50_on(IDX)
E['grp'] = np.where(E['in0050'], '0050', np.where(E['att'] == 1, 'low', np.where(E['att'] == 2, 'mid', 'high')))


def season_t(df, col):
    s = df.groupby('k')[col].mean()
    return {'mean': float(s.mean()), 't_season': float(tstat(s)), 'n_seasons': int(s.notna().sum()),
            't_date': float(tstat(df.groupby('A')[col].mean())), 'n_events': int(df[col].notna().sum())}


out = {'sample': {'events': int(len(E)), 'seasons': int(E['k'].nunique()), 'lowatt': int(E['lowatt'].sum()),
                  'tpex': int((E['mkt'] == 'tpex').sum()), 'A_method': E['A_method'].value_counts().to_dict()}}
for H in [20, 40, 60, 120]:
    col = f'r{H}_j0'
    E[f'ab{H}'] = E[col] - E.groupby(['k', 'grp'])[col].transform('mean')
    ok = (E['e_j0'] >= 1) & (E['e_j0'] + H - 1 < len(IDX))
    b = np.full(len(E), np.nan); ee = E['e_j0'].to_numpy()
    b[ok.to_numpy()] = e50[(ee + H - 1)[ok.to_numpy()]] / e50[(ee - 1)[ok.to_numpy()]] - 1
    E[f'b50_{H}'] = b
out['P1'] = {}
for sig in ['SUEP', 'SUE']:
    for g in ['low', 'mid', 'high', '0050', 'all']:
        mm = (E['grp'] == g) if g != 'all' else pd.Series(True, index=E.index)
        top = E[mm & E[f'top_{sig}']]; bot = E[mm & E[f'bot_{sig}']]
        out['P1'][f'{sig}|{g}|H60'] = {'top_ab': season_t(top, 'ab60'), 'bot_ab': season_t(bot, 'ab60'),
                                       'top_vs50_net': season_t(top.assign(x=top['r60_j0'] - top['b50_60'] - 0.006), 'x')}
    a = E[(E['grp'] == 'low') & E[f'top_{sig}']].groupby('k')['ab60'].mean()
    b = E[(E['grp'].isin(['high', '0050'])) & E[f'top_{sig}']].groupby('k')['ab60'].mean()
    d = (a - b).dropna(); out[f'P1_low_minus_high_{sig}'] = {'mean': float(d.mean()), 't_season': float(tstat(d)), 'n': int(len(d))}
    a = E[(E['grp'] == 'low') & E[f'top_{sig}']].groupby('k')['r60_j0'].mean()
    b = E[(E['grp'] == 'low') & E[f'bot_{sig}']].groupby('k')['r60_j0'].mean()
    d = (a - b).dropna(); out[f'low_top_minus_bot_{sig}'] = {'mean': float(d.mean()), 't_season': float(tstat(d))}
    a = E[E[f'top_{sig}']].groupby('k')['r60_j0'].mean(); b = E[E[f'bot_{sig}']].groupby('k')['r60_j0'].mean()
    d = (a - b).dropna(); out[f'all_top_minus_bot_{sig}'] = {'mean': float(d.mean()), 't_season': float(tstat(d))}


def fm(df, xs):
    coefs = []
    for k, g in df.groupby('k'):
        g = g.dropna(subset=xs + ['r60_j0'])
        if len(g) < 30:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[x].rank(pct=True).to_numpy() - 0.5 for x in xs])
        y = g['r60_j0'].to_numpy() - g['r60_j0'].mean()
        coefs.append(np.linalg.lstsq(X, y, rcond=None)[0][1:])
    C = np.array(coefs)
    return {x: {'coef': float(C[:, i].mean()), 't': float(tstat(C[:, i]))} for i, x in enumerate(xs)} | {'n_seasons': len(C)}


E['lsize'] = np.log(E['mcap']); E['lv60'] = np.log(E['V60'])
xs = ['SUEP', 'prev3', 'p52', 'REVSUE', 'lsize', 'lv60']
out['P2_FM_all'] = fm(E, xs); out['P2_FM_low'] = fm(E[E['grp'] == 'low'], xs); out['P2_FM_high'] = fm(E[E['grp'].isin(['high', '0050'])], xs)
# 時代：2019Q2～2023Q4 vs 2024Q1 起
for nm, mm in [('VAL', E['k'] <= 2023 * 4 + 3), ('HOLD', E['k'] >= 2024 * 4)]:
    out[f'P2_FM_all_{nm}'] = fm(E[mm], xs)
    t = E[mm & E['top_SUEP'] & ~E['in0050']]
    out[f'top_nonfifty_ab60_{nm}'] = season_t(t, 'ab60')
    out[f'top_nonfifty_vs50net_{nm}'] = season_t(t.assign(x=t['r60_j0'] - t['b50_60'] - 0.006), 'x')
# P3
lt = E[(E['grp'] == 'low') & E['top_SUEP']].copy()
lt['busy'] = lt['n_same_day'] >= lt.groupby('k')['n_same_day'].transform(lambda s: s.quantile(0.8))
for f in ['last2', 'busy']:
    a = lt[lt[f]].groupby('k')['ab60'].mean(); b = lt[~lt[f]].groupby('k')['ab60'].mean(); d = (a - b).dropna()
    out[f'P3_{f}'] = {'mean_diff': float(d.mean()), 't_season': float(tstat(d)), 'n_yes': int(lt[f].sum())}
out['P3_by_q_low_top'] = {int(q): season_t(g, 'ab60') for q, g in lt.groupby('q')}
# P4 曲線（非 0050 top − 非 0050 全體）
nf = ~E['in0050'].to_numpy()
for nm, msk in [('low', (E['grp'] == 'low').to_numpy()), ('nonfifty', nf), ('high', (E['grp'] == 'high').to_numpy())]:
    mt = msk & E['top_SUEP'].to_numpy()
    out[f'P4_curve_{nm}_top_minus_grp'] = [round(float(np.nanmean(PATH[mt, h]) - np.nanmean(PATH[msk, h])), 5) for h in range(0, 121, 10)]
# P5 j 衰減
jd = {}
for j in range(10):
    col = f'r60_j{j}'
    ab = E[col] - E.groupby(['k', 'grp'])[col].transform('mean')
    for nm, mm in [('low', (E['grp'] == 'low') & E['top_SUEP']), ('nonfifty', ~E['in0050'] & E['top_SUEP'])]:
        jd.setdefault(nm, {})[j] = season_t(E[mm].assign(x=ab[mm]), 'x')['mean']
out['j_decay_ab60'] = jd
# 分市場、分年、分產業、去掉最大貢獻股（非 0050 top，ab60）
t = E[~E['in0050'] & E['top_SUEP']].copy()
out['by_market'] = {m_: season_t(g, 'ab60') for m_, g in t.groupby('mkt')}
t['yr'] = t['A'].dt.year
out['by_year_ab60'] = {int(y): float(g['ab60'].mean()) for y, g in t.groupby('yr')}
out['by_year_vs50net'] = {int(y): float((g['r60_j0'] - g['b50_60'] - 0.006).mean()) for y, g in t.groupby('yr')}
rv = pd.read_parquet(DATA / 'rev_long.parquet'); indmap = rv.sort_values('ym').groupby('code')['ind'].last()
t['ind'] = t['code'].map(indmap).fillna('NA')
bi = t.groupby('ind')['ab60'].agg(['mean', 'count']).sort_values('count', ascending=False).head(12)
out['by_industry_ab60'] = {k: {'mean': float(r['mean']), 'n': int(r['count'])} for k, r in bi.iterrows()}
contrib = t.groupby('code')['ab60'].sum().sort_values()
for n in [1, 5, 10]:
    tt = t[~t['code'].isin(contrib.index[-n:])]
    out[f'drop_top{n}_contrib_codes'] = season_t(tt, 'ab60')
out['top_nonfifty_ab60_all'] = season_t(t, 'ab60')
out['top_nonfifty_vs50net_all'] = season_t(t.assign(x=t['r60_j0'] - t['b50_60'] - 0.006), 'x')
out['top_nonfifty_raw_mean_r60'] = float(t['r60_j0'].mean()); out['nonfifty_all_mean_r60'] = float(E.loc[~E['in0050'], 'r60_j0'].mean())
R['EVENT_OOS'] = out
json.dump(R, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=float)
for k in ['sample', 'P1_low_minus_high_SUEP', 'low_top_minus_bot_SUEP', 'all_top_minus_bot_SUEP', 'all_top_minus_bot_SUE', 'P2_FM_all', 'P2_FM_low', 'P2_FM_high',
          'P2_FM_all_VAL', 'P2_FM_all_HOLD', 'top_nonfifty_ab60_VAL', 'top_nonfifty_ab60_HOLD', 'top_nonfifty_vs50net_VAL', 'top_nonfifty_vs50net_HOLD',
          'P3_last2', 'P3_busy', 'P4_curve_low_top_minus_grp', 'P4_curve_nonfifty_top_minus_grp', 'P4_curve_high_top_minus_grp', 'j_decay_ab60', 'by_market',
          'by_year_ab60', 'by_year_vs50net', 'drop_top5_contrib_codes', 'top_nonfifty_ab60_all', 'top_nonfifty_vs50net_all', 'top_nonfifty_raw_mean_r60', 'nonfifty_all_mean_r60']:
    print(k, json.dumps(out[k], ensure_ascii=False, default=float)[:700])
for g in ['low', 'mid', 'high', '0050']:
    print(g, out['P1'][f'SUEP|{g}|H60']['top_ab'])
