#!/usr/bin/env python3
"""探索 (2)(3)(4)：首次公告值（新聞）與面板（更正後最新值）的比較。只看資料與分數排名，不看任何報酬。
來源：o_v4 的 raw_news_mentions.parquet（鉅亨新聞 2013～2021，文章日期＝可得時點）。
輸出 data/fv_cells.csv（每個 code×ym 的新聞彙整與面板比較）、data/top30_cells.csv、explore_fv.json。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v5lib as L
L4, K, V = L.L4, L.K, L.V

L4.check_freeze(); L4.setup_px()
YFX, Y5F = L4.load_rfix()
MEN = pd.read_parquet(L.V4DIR / 'raw_news_mentions.parquet')
# 最新值水準 L0（與 o_v4 build_rfix 同優先序：tw_revenue sii > taiwanan > rev_5y）
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str}); d = d[d.market == 'sii']; d['ym'] = d.month.str[:7]
T = pd.read_csv(L4.V3_INPUTS['src_taiwanan'][0], dtype={'code': str})
r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str})
L0 = {}
for c, y, v in zip(r5.code, r5.ym, r5.revenue): L0[(c, y)] = float(v)
for c, y, v in zip(T.code, T.ym, T.rev): L0[(c, y)] = float(v)
for c, y, v in zip(d.code, d.ym, d.revenue_k_twd): L0[(c, y)] = float(v)
SRC = {}
for c, y in zip(r5.code, r5.ym): SRC[(c, y)] = 'rev5y'
for c, y in zip(T.code, T.ym): SRC[(c, y)] = 'taiwanan'
for c, y in zip(d.code, d.ym): SRC[(c, y)] = 'tw_revenue'

Z = MEN[(MEN.lag <= 1) & (MEN.ym >= '2013-01')].copy()
Z['lv'] = [L0.get((c, y), np.nan) for c, y in zip(Z.code, Z.ym)]
Z['rel'] = Z.rev_k / Z.lv - 1
Z['date'] = pd.to_datetime(Z.date)
# 每則對面板：精度 ≤ 0.25% 者才可判一致與否
Zp = Z[(Z.prec <= 0.0025) & Z.lv.notna() & (Z.lv > 0)]
g = Zp.groupby(['code', 'ym'])
A = pd.DataFrame({'first_date': g.date.min(), 'n': g.size(), 'med': g.rev_k.median(), 'lo': g.rev_k.min(), 'hi': g.rev_k.max(),
                  'lv': g.lv.first(), 'yoy_news': g.yoy_news.median(), 'mom_news': g.mom_news.median()}).reset_index()
# 最早一則的值（首次值候選）
e = Zp.sort_values(['code', 'ym', 'date']).groupby(['code', 'ym']).first()
A = A.merge(e[['rev_k']].rename(columns={'rev_k': 'earliest'}).reset_index(), on=['code', 'ym'])
A['spread'] = A.hi / A.lo - 1
A['rel_med'] = A.med / A.lv - 1
A['rel_first'] = A.earliest / A.lv - 1
A['agree'] = A.rel_first.abs() < 0.005
A['src'] = [SRC.get((c, y), '') for c, y in zip(A.code, A.ym)]
def sh(ym, k): return str(pd.Period(ym, 'M') - k)
# 內部一致：新聞同句年增／月增率與面板上月／去年同月水準換算差 ≤ 1pp（以首次值 earliest 計算）
ic = []
for c, y, v, yo, mo in zip(A.code, A.ym, A.earliest, A.yoy_news, A.mom_news):
    b12, b1 = L0.get((c, sh(y, 12))), L0.get((c, sh(y, 1)))
    ok_y = bool(b12 and np.isfinite(yo) and abs((v / b12 - 1) * 100 - yo) <= 1)
    ok_m = bool(b1 and np.isfinite(mo) and abs((v / b1 - 1) * 100 - mo) <= 1)
    ic.append(ok_y or ok_m)
A['internal_ok'] = ic
A['consensus'] = (A.n >= 2) & (A.spread <= 0.005)
A.to_csv(HERE / 'data' / 'fv_cells.csv', index=False)

# 分數前 30 名股月（V4 非 AB、R_FIX；只看分數，不看報酬）→ 用到的本期水準格 m−1..m−3
S = L4.V4(YFX, Y5F, L4.disp_mask_fn('trade'))
rows = []
for dd in S.sig_days:
    cs = S.cross_section(dd)
    o = V.V2.order(cs, 'official')[:min(30, int(cs['el'].sum()))]
    m = pd.Period(dd, 'M')
    for r, j in enumerate(o):
        for k in (1, 2, 3):
            rows.append((str(dd.date()), K.COLS[j], r + 1, str(m - k), 'cur'))
            rows.append((str(dd.date()), K.COLS[j], r + 1, str(m - k - 12), 'base'))
T30 = pd.DataFrame(rows, columns=['sig', 'code', 'rank', 'ym', 'role'])
T30.to_csv(HERE / 'data' / 'top30_cells.csv', index=False)
Akey = A.set_index(['code', 'ym'])
def stat(mask):
    x = A[mask]
    return {'n_cells': int(len(x)), 'disagree_rate': float((~x.agree).mean()) if len(x) else None, 'n_disagree': int((~x.agree).sum()),
            'disagree_rate_consensus': float((~x[x.consensus].agree).mean()) if x.consensus.any() else None,
            'n_disagree_consensus_or_internal': int((~x.agree & (x.consensus | x.internal_ok)).sum()),
            'rate_disagree_consensus_or_internal': float((~x.agree & (x.consensus | x.internal_ok)).mean()) if len(x) else None}
t30k = set(zip(T30.code, T30.ym))
t30cur = set(zip(T30[T30.role == 'cur'].code, T30[T30.role == 'cur'].ym))
t10cur = set(zip(T30[(T30.role == 'cur') & (T30['rank'] <= 10)].code, T30[(T30.role == 'cur') & (T30['rank'] <= 10)].ym))
inA = [(c, y) in t30cur for c, y in zip(A.code, A.ym)]
in10 = [(c, y) in t10cur for c, y in zip(A.code, A.ym)]
out = {'n_news_cells_prec_ok': int(len(A)), 'all': stat(np.ones(len(A), bool)), 'top30_cur': stat(np.array(inA)), 'top10_cur': stat(np.array(in10))}
a, b = out['top30_cur']['disagree_rate'], out['all']['disagree_rate']
out['ratio_top30_vs_all'] = a / b if b else None
# 首次值覆蓋：前 30 名股月（訊號日 d）中，本期 3 格與基期 3 格都有「d 之前的新聞首次值」的比例
fd = dict(zip(zip(A.code, A.ym), A.first_date))
T30['has_fv'] = [((c, y) in fd) and fd[(c, y)] < pd.Timestamp(s) for c, y, s in zip(T30.code, T30.ym, T30.sig)]
cov = T30.groupby(['sig', 'code']).has_fv.agg(['all', 'any', 'mean'])
cur = T30[T30.role == 'cur'].groupby(['sig', 'code']).has_fv.all()
out['top30_stockmonths'] = int(len(cov))
out['top30_cov_all6'] = float(cov['all'].mean()); out['top30_cov_cur3'] = float(cur.mean())
y13 = cov.reset_index(); y13 = y13[(y13.sig >= '2013-03') & (y13.sig <= '2021-12-31')]
out['top30_cov_all6_2013_2021'] = float(y13['all'].mean())
cur2 = cur.reset_index(); cur2 = cur2[(cur2.sig >= '2013-03') & (cur2.sig <= '2021-12-31')]
out['top30_cov_cur3_2013_2021'] = float(cur2['has_fv'].mean())
# 分布：年份、方向、來源、產業
dis = A[~A.agree].copy()
dis['dir'] = np.where(dis.rel_first < 0, 'first<latest（事後上修）', 'first>latest（事後下修）')
ind = V.SL['industry'] if 'industry' in V.SL else None
dis['industry'] = [V.SL['industry'].get(c, '（無）') if c in V.SL.index else '（已下市）' for c in dis.code]
A['industry'] = [V.SL['industry'].get(c, '（無）') if c in V.SL.index else '（已下市）' for c in A.code]
out['disagree_by_year'] = {y: {'n': int(len(g_)), 'rate': float((~g_.agree).mean())} for y, g_ in A.groupby(A.ym.str[:4])}
out['disagree_by_dir'] = dis.dir.value_counts().to_dict()
out['disagree_abs_rel_quantiles'] = {str(q): float(dis.rel_first.abs().quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
out['disagree_by_src'] = {s: {'n': int(len(g_)), 'rate': float((~g_.agree).mean())} for s, g_ in A.groupby('src')}
el = A.industry.isin(['半導體業', '電子零組件業', '其他電子業', '光電業', '通信網路業', '電腦及週邊設備業', '電子通路業', '資訊服務業'])
out['disagree_electronics'] = {'n': int(el.sum()), 'rate': float((~A[el].agree).mean())}
out['disagree_non_electronics'] = {'n': int((~el).sum()), 'rate': float((~A[~el].agree).mean())}
out['disagree_by_industry_top'] = {i: {'n': int(len(g_)), 'rate': float((~g_.agree).mean())} for i, g_ in A.groupby('industry') if len(g_) >= 150}
# 排名分層
rk = T30[T30.role == 'cur'].groupby(['code', 'ym'])['rank'].min()
A['best_rank'] = [rk.get((c, y), np.nan) for c, y in zip(A.code, A.ym)]
for lo, hi in ((1, 10), (11, 30)):
    m_ = (A.best_rank >= lo) & (A.best_rank <= hi)
    out[f'disagree_rank_{lo}_{hi}'] = {'n': int(m_.sum()), 'rate': float((~A[m_].agree).mean()) if m_.any() else None}
out['examples_disagree_top10'] = A[(~A.agree) & (A.best_rank <= 10)][['code', 'ym', 'earliest', 'lv', 'rel_first', 'n', 'consensus', 'internal_ok', 'first_date']].astype(str).to_dict('records')[:40]
json.dump(out, open(HERE / 'explore_fv.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != 'examples_disagree_top10'}, ensure_ascii=False, indent=1, default=str))
