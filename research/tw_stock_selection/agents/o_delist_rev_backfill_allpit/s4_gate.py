#!/usr/bin/env python3
"""步驟 2(c) 品質閘門（只比資料，不看任何報酬）：新聞值與 taiwanan 值在「同期存活公司」上與 tw_revenue 比對。
新聞水準：|相對差| ≤ 0.6% 的比例 ≥ 98%（以億元 0.01 計）；新聞年增：|差| ≤ 0.1pp 的比例 ≥ 98%（只比 2014 起，避開 2013 IFRS 斷點）。
taiwanan：水準 |相對差| ≤ 0.1% 的比例 ≥ 98%（當月值、上月值、去年同月值分開）。依年份分期；不合格的（來源×類型×年）整段剔除。
python3 -I -B s4_gate.py <news_ver> <tw_ver>   輸出 gate_<news_ver>_<tw_ver>.json"""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
NV = sys.argv[1] if len(sys.argv) > 1 else 'P1'; TV = sys.argv[2] if len(sys.argv) > 2 else 'T1'
sys.path.insert(0, str(HERE)); import bflib as B; sys.path.pop(0)
LV = B.tw_levels()                         # tw_revenue 水準（PeriodIndex × code，千元）
def lv(c, ym):
    try: return LV.at[pd.Period(ym, 'M'), c]
    except KeyError: return np.nan
out = {'news_ver': NV, 'tw_ver': TV, 'news': {}, 'tw': {}}
# ---- 新聞
N = pd.read_parquet(HERE / f'data/news_vals_{NV}.parquet')
N['tw'] = [lv(c, y) for c, y in zip(N.code, N.ym)]
N['rel'] = N.rev_k / N.tw - 1
N['yr'] = N.ym.str[:4]
P = pd.PeriodIndex(N.ym, freq='M')
N['tw_ly'] = [lv(c, str(p - 12)) for c, p in zip(N.code, P)]
N['yoy_tw'] = (N.tw / N.tw_ly - 1) * 100
ov = N[np.isfinite(N.tw) & (N.tw > 0)]
def rate(x, thr): return dict(n=int(len(x)), ok=float((x.abs() <= thr).mean()) if len(x) else None)
out['news']['level_all'] = rate(ov.rel, 0.006)
out['news']['level_by_year'] = {y: rate(g.rel, 0.006) for y, g in ov.groupby('yr')}
out['news']['level_by_basis'] = {b: rate(g.rel, 0.006) for b, g in ov.groupby('basis')}
out['news']['level_by_ndec'] = {str(b): rate(g.rel, 0.006) for b, g in ov.groupby('ndec')}
out['news']['level_by_how'] = {str(b): rate(g.rel, 0.006) for b, g in ov.groupby('how')}
oy = ov[np.isfinite(ov.yoy) & np.isfinite(ov.yoy_tw) & (ov.ym >= '2014-01')]
out['news']['yoy_all'] = rate(oy.yoy - oy.yoy_tw, 0.1)
out['news']['yoy_by_year'] = {y: rate(g.yoy - g.yoy_tw, 0.1) for y, g in oy.groupby('yr')}
out['news']['n_overlap_codes'] = int(ov.code.nunique()); out['news']['n_nonoverlap'] = int((~np.isfinite(N.tw)).sum())
bad = ov[ov.rel.abs() > 0.006].copy(); bad['absrel'] = bad.rel.abs()
out['news']['bad_examples'] = bad.sort_values('absrel').tail(25)[['code', 'ym', 'rev_k', 'tw', 'rel', 'basis', 'ndec', 'seg']].astype(str).to_dict('records')
out['news']['bad_mid_examples'] = bad[(bad.absrel < 0.05)].head(25)[['code', 'ym', 'rev_k', 'tw', 'rel', 'basis', 'ndec', 'seg']].astype(str).to_dict('records')
# ---- taiwanan
T = B.taiwanan(TV)
for fld, sh in (('rev', 0), ('prev', 1), ('ly', 12)):
    x = T[np.isfinite(T[fld])].copy(); pp = pd.PeriodIndex(x.ym, freq='M') - sh
    x['tw'] = [lv(c, str(p)) for c, p in zip(x.code, pp)]; x = x[np.isfinite(x.tw) & (x.tw > 0)]
    x['rel'] = x[fld] / x.tw - 1; x['yr'] = [str(p.year) for p in pp[np.isin(np.arange(len(pp)), np.arange(len(pp)))]][:len(x)] if False else [str((pd.Period(y, 'M') - sh).year) for y in x.ym]
    out['tw'][fld] = dict(all=rate(x.rel, 0.001), by_year={y: rate(g.rel, 0.001) for y, g in x.groupby('yr')})
(HERE / f'gate_{NV}_{TV}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: (v if k != 'news' else {kk: vv for kk, vv in v.items() if 'examples' not in kk}) for k, v in out.items()}, ensure_ascii=False, indent=0)[:5000])
