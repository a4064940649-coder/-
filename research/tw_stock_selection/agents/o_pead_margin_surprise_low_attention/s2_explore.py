#!/usr/bin/env python3
"""第 2 步：探索（只用挑選期：公告季 2016Q1～2019Q1；SUE⊥ 從 2016Q1 起才有 PIT 百分位）。輸出 explore.json。
所有 t 值以「財報季」分群（每季先平均，再對季做 t；13 季），另報以公告日分群的 t。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pcore import DATA, HERE, SEL_END_Q, load_panel, e50_on, tstat, FIN_IND
from pengine import Market, run, perf

KSEL = SEL_END_Q[0] * 4 + SEL_END_Q[1] - 1
E = pd.read_parquet(DATA / 'events.parquet')
paths = np.load(DATA / 'event_paths.npz')
PATH, EWP = paths['PATH'], paths['EWP']
sel = (E['k'] <= KSEL) & E['SUEP'].notna()
E = E[sel].copy(); PATH = PATH[sel.to_numpy()]; EWP = EWP[sel.to_numpy()]
assert E['k'].max() <= KSEL
P = load_panel(); IDX = pd.DatetimeIndex(P['IDX'])
e50 = e50_on(IDX)
out = {'sample': {'events': int(len(E)), 'seasons': int(E['k'].nunique()), 'k_range': [int(E['k'].min()), int(E['k'].max())],
                  'lowatt': int(E['lowatt'].sum()), 'tpex': int((E['mkt'] == 'tpex').sum()),
                  'A_method': E['A_method'].value_counts().to_dict(), 'late_share': float(E['late'].mean())}}
# 0050 同窗報酬（j0 進場前收→出場前收；近似）
def r50(e, H):
    ok = (e >= 1) & (e + H - 1 < len(IDX))
    r = np.full(len(e), np.nan)
    r[ok] = e50[(e + H - 1)[ok]] / e50[(e - 1)[ok]] - 1
    return r
for H in [20, 40, 60, 120]:
    E[f'b50_{H}'] = r50(E['e_j0'].to_numpy(), H)
E['grp'] = np.where(E['in0050'], '0050', np.where(E['att'] == 1, 'low', np.where(E['att'] == 2, 'mid', 'high')))
E['dec'] = E.groupby('k')['SUEP'].transform(lambda s: np.ceil(s.rank(pct=True) * 10).clip(1, 10))


def season_t(df, col):
    s = df.groupby('k')[col].mean()
    return {'mean': float(s.mean()), 't_season': float(tstat(s)), 'n_seasons': int(s.notna().sum()),
            't_date': float(tstat(df.groupby('A')[col].mean())), 'n_events': int(df[col].notna().sum())}


# ① 十分位 × 關注度：CAR（相對同池等權）20/40/60/120；以及 top1/5 − 同組平均
car = {}
AR = PATH - EWP
for g in ['low', 'mid', 'high', '0050']:
    m = (E['grp'] == g).to_numpy()
    car[g] = {}
    for d in range(1, 11):
        md = m & (E['dec'] == d).to_numpy()
        car[g][d] = {h: float(np.nanmean(AR[md, h])) for h in [5, 20, 40, 60, 120]}
out['car_vs_pool_ew_by_decile'] = car
# 曲線：低關注 top 1/5 − 低關注全體（0～120 日）
curves = {}
for g in ['low', 'mid', 'high', '0050']:
    m = (E['grp'] == g).to_numpy(); mt = m & E['top_SUEP'].to_numpy(); mb = m & E['bot_SUEP'].to_numpy()
    curves[g] = {'top_minus_grp': [round(float(np.nanmean(PATH[mt, h]) - np.nanmean(PATH[m, h])), 5) for h in range(0, 121, 5)],
                 'bot_minus_grp': [round(float(np.nanmean(PATH[mb, h]) - np.nanmean(PATH[m, h])), 5) for h in range(0, 121, 5)],
                 'n_top': int(mt.sum()), 'n_grp': int(m.sum())}
out['curves_days_0_120_step5'] = curves
# ② P1：r60 j0（開盤到開盤）top − 同組同季平均（毛）；top 對 0050（扣約 0.6% 來回）
p1 = {}
for H in [20, 40, 60, 120]:
    col = f'r{H}_j0'
    E[f'ab{H}'] = E[col] - E.groupby(['k', 'grp'])[col].transform('mean')
    E[f'ab{H}_s'] = E[col] - E.groupby(['k', 'grp'])[col].transform('mean')
    for sig in ['SUEP', 'SUE']:
        for g in ['low', 'mid', 'high', '0050', 'all']:
            mm = (E['grp'] == g) if g != 'all' else pd.Series(True, index=E.index)
            top = E[mm & E[f'top_{sig}']]
            bot = E[mm & E[f'bot_{sig}']]
            p1[f'{sig}|{g}|H{H}'] = {'top_ab': season_t(top, f'ab{H}'), 'bot_ab': season_t(bot, f'ab{H}'),
                                     'top_vs50_net': season_t(top.assign(x=top[col] - top[f'b50_{H}'] - 0.006), 'x')}
out['P1'] = p1
# 組間差：低關注 top_ab − 高關注 top_ab（季層級）
for sig in ['SUEP', 'SUE']:
    a = E[(E['grp'] == 'low') & E[f'top_{sig}']].groupby('k')['ab60'].mean()
    b = E[(E['grp'].isin(['high', '0050'])) & E[f'top_{sig}']].groupby('k')['ab60'].mean()
    d = (a - b).dropna()
    out[f'P1_low_minus_high_{sig}'] = {'mean': float(d.mean()), 't_season': float(tstat(d)), 'n': int(len(d))}
# 多空：低關注 top − bot
for sig in ['SUEP', 'SUE']:
    a = E[(E['grp'] == 'low') & E[f'top_{sig}']].groupby('k')['r60_j0'].mean()
    b = E[(E['grp'] == 'low') & E[f'bot_{sig}']].groupby('k')['r60_j0'].mean()
    d = (a - b).dropna()
    out[f'low_top_minus_bot_{sig}'] = {'mean': float(d.mean()), 't_season': float(tstat(d))}
# ③ P2 Fama-MacBeth（每季橫斷面 OLS；變數為季內百分位，y＝r60_j0 − 季平均）
def fm(df, xs):
    coefs = []
    for k, g in df.groupby('k'):
        g = g.dropna(subset=xs + ['r60_j0'])
        if len(g) < 30:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[x].rank(pct=True).to_numpy() - 0.5 for x in xs])
        y = g['r60_j0'].to_numpy() - g['r60_j0'].mean()
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        coefs.append(b[1:])
    C = np.array(coefs)
    return {x: {'coef': float(C[:, i].mean()), 't': float(tstat(C[:, i]))} for i, x in enumerate(xs)} | {'n_seasons': len(C)}
E['lsize'] = np.log(E['mcap']); E['lv60'] = np.log(E['V60'])
xs = ['SUEP', 'prev3', 'p52', 'REVSUE', 'lsize', 'lv60']
out['P2_FM_all'] = fm(E, xs)
out['P2_FM_low'] = fm(E[E['grp'] == 'low'], xs)
out['P2_FM_high'] = fm(E[E['grp'].isin(['high', '0050'])], xs)
out['P2_FM_all_SUEraw'] = fm(E, ['SUE', 'prev3', 'p52', 'REVSUE', 'lsize', 'lv60'])
# ④ P3 注意力分散（低關注 top 事件）
lt = E[(E['grp'] == 'low') & E['top_SUEP']].copy()
lt['busy'] = lt['n_same_day'] >= lt.groupby('k')['n_same_day'].transform(lambda s: s.quantile(0.8))
for f in ['last2', 'busy']:
    a = lt[lt[f]].groupby('k')['ab60'].mean(); b = lt[~lt[f]].groupby('k')['ab60'].mean()
    d = (a - b).dropna()
    out[f'P3_{f}'] = {'mean_diff': float(d.mean()), 't_season': float(tstat(d)), 'n_yes': int(lt[f].sum()), 'n_no': int((~lt[f]).sum())}
out['P3_by_q'] = {int(q): season_t(g, 'ab60') for q, g in lt.groupby('q')}
# ⑤ j 衰減：低關注 top，r60_j{0..9} − 同組同季平均 r60_j{jd}
jdec = {}
for jd in range(10):
    col = f'r60_j{jd}'
    ab = E[col] - E.groupby(['k', 'grp'])[col].transform('mean')
    m = (E['grp'] == 'low') & E['top_SUEP']
    jdec[jd] = season_t(E[m].assign(x=ab[m]), 'x')['mean']
out['j_decay_low_top_ab60'] = jdec
# 市場別、Q4、A 方法
out['by_market_low_top_ab60'] = {m: season_t(g, 'ab60') for m, g in E[(E['grp'] == 'low') & E['top_SUEP']].groupby('mkt')}
out['by_Amethod_low_top_ab60'] = {m: season_t(g, 'ab60') for m, g in E[(E['grp'] == 'low') & E['top_SUEP']].groupby('A_method')}
# ⑥ 成本：低關注 top 事件在 100萬／300萬／1000萬（20 槽）的參與率 Q/OA 分布（OA＝0.03×V60）
part = {}
for K in [1e6, 3e6, 1e7]:
    q = (K / 20) / (0.03 * lt['V60'])
    part[int(K)] = {p: float(q.quantile(p)) for p in [0.1, 0.5, 0.9]}
    imp = 0.001 + 0.5 * lt['sig60'].fillna(0.03) * np.sqrt(q.clip(upper=0.2 * 1.0 / 1.0))
    part[int(K)]['impact_median_oneside'] = float(imp.median())
out['participation_low_top'] = part
# ⑦ 診斷：毛利率／營益率變動與 SUE⊥ 的相關（機制：SUE⊥ 是否代表利潤率驚喜）
out['corr_SUEP_dGM'] = float(E[['SUEP', 'dGM']].corr(method='spearman').iloc[0, 1])
out['corr_SUEP_dOM'] = float(E[['SUEP', 'dOM']].corr(method='spearman').iloc[0, 1])
out['corr_SUEP_REVSUE'] = float(E[['SUEP', 'REVSUE']].corr(method='spearman').iloc[0, 1])
out['corr_SUE_REVSUE'] = float(E[['SUE', 'REVSUE']].corr(method='spearman').iloc[0, 1])
out['corr_SUEP_prev3'] = float(E[['SUEP', 'prev3']].corr(method='spearman').iloc[0, 1])
out['corr_SUEP_p52'] = float(E[['SUEP', 'p52']].corr(method='spearman').iloc[0, 1])
# 前視檢查：PIT SUE⊥ 與同季 SUE⊥（含較晚公告者）一致性
E['SUEP_c'] = E['SUE_pct_c'] - E.groupby('k').apply(lambda g: pd.Series(np.polyval(np.polyfit(g['REV_pct_c'].fillna(0.5), g['SUE_pct_c'], 1), g['REV_pct_c'].fillna(0.5)), index=g.index)).reset_index(level=0, drop=True)
out['corr_SUEP_pit_vs_contemp'] = float(E[['SUEP', 'SUEP_c']].corr(method='spearman').iloc[0, 1])
# 下市公司事件
last = pd.DataFrame(P['C']).iloc[-20:].notna().any().to_numpy()
COLS = np.array(P['COLS'])
gone = set(COLS[~last])
out['events_of_later_delisted'] = int(E['code'].isin(gone).sum())
json.dump(out, open(HERE / 'explore.json', 'w'), ensure_ascii=False, indent=1, default=float)
pr = lambda k: print(k, json.dumps(out[k], ensure_ascii=False, default=float)[:900])
for k in ['sample', 'P1_low_minus_high_SUEP', 'P1_low_minus_high_SUE', 'low_top_minus_bot_SUEP', 'low_top_minus_bot_SUE', 'P2_FM_all', 'P2_FM_low',
          'P2_FM_high', 'P3_last2', 'P3_busy', 'P3_by_q', 'j_decay_low_top_ab60', 'by_market_low_top_ab60', 'participation_low_top',
          'corr_SUEP_dGM', 'corr_SUEP_dOM', 'corr_SUEP_REVSUE', 'corr_SUE_REVSUE', 'corr_SUEP_pit_vs_contemp', 'events_of_later_delisted']:
    pr(k)
for g in ['low', 'mid', 'high', '0050']:
    print(g, 'H60 top_ab', out['P1'][f'SUEP|{g}|H60']['top_ab'], 'vs50net', out['P1'][f'SUEP|{g}|H60']['top_vs50_net'])
print('curves low top', out['curves_days_0_120_step5']['low']['top_minus_grp'])
print('curves high top', out['curves_days_0_120_step5']['high']['top_minus_grp'])
