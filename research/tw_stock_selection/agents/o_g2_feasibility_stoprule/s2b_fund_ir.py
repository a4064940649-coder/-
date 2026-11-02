#!/usr/bin/env python3
"""探索（2）基金：SITCA 月度績效評比 → 國內股票型主動基金逐月報酬（一個月報酬；沒有時用「自今年以來」差分），
對 0050 含息（finlab etf_0050.csv 月底）月超額 IR、t、分段；存活者偏誤上下界。
可得月份：2006-08～2026-09（2004-01～2006-07 為舊版面，解析器未支援，記入限制）。
輸出：data/fund_monthly.parquet、data/fund_ir.parquet、explore.json['funds']、results.json['FUNDS']。"""
import json, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__file__))
import g2lib as G

G.check_prereg()
d = pd.read_parquet(G.DATA / 'fund_panel_long_all.parquet')
meta = json.loads((G.DATA / 'fund_parse_meta_all.json').read_text())
p1 = d[d.part == 1].copy(); p2 = d[d.part == 2].copy()
p1['catn'] = p1.cat.str.replace('、', '.', regex=False)
dom = p1[p1.catn.str.contains('股票型') & p1.catn.str.contains('投資國內') & ~p1.catn.str.contains('平衡')].copy()
dom['sub'] = dom.catn.str.split('|').str[-1]

def subclass(s):
    if '一般股票型' in s: return 'general'
    if '科技類' in s: return 'tech'
    if '中小型' in s: return 'smallmid'
    if '價值型' in s: return 'value'
    if '上櫃' in s: return 'otc'
    if '中概' in s: return 'china'
    if '指數' in s or 'ETF' in s or '主動式交易所' in s: return 'index_etf'
    if '特殊' in s: return 'special'
    return 'parent'
dom['cls'] = dom['sub'].map(subclass)
# 份額類別排除：TISA、N、B（配息）、外幣、ETF、指數
EXCL = re.compile(r'(TISA|ETF|指數|N-|N類|-N|B類|-B|配息|美元|USD|人民幣|RMB|月配|季配|半年配|-I|I類|S類|-S$|期貨|槓桿|反向)')
dom['excl_name'] = dom.name.str.contains(EXCL)
# 每月每檔：分類取最具體（非 parent），報酬欄同一檔在 parent 與子類重複，取第一筆
lab = dom[dom.cls != 'parent'].groupby(['ym', 'name']).cls.first()
val = dom.groupby(['ym', 'name'])[['m1', 'ytd']].first()
P = val.join(lab, how='left').reset_index()
P['cls'] = P.cls.fillna('parent')
# 成立日（第二部份）
inc = p2[p2.inc.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$')].groupby(['ym', 'name']).inc.first()
P = P.merge(inc.reset_index(), on=['ym', 'name'], how='left')
P['date'] = pd.to_datetime(P.ym, format='%Y%m') + pd.offsets.MonthEnd(0)
P = P.sort_values(['name', 'date'])

# 月報酬：一個月報酬（有就用）；否則 YTD 差分（1 月＝YTD）
P['prev_ytd'] = P.groupby('name').ytd.shift(1); P['prev_date'] = P.groupby('name').date.shift(1)
consec = (P.prev_date + pd.offsets.MonthEnd(1)) == P.date
r_ytd = np.where(P.date.dt.month == 1, P.ytd / 100, np.where(consec, (1 + P.ytd / 100) / (1 + P.prev_ytd / 100) - 1, np.nan))
P['r_ytd'] = r_ytd
P['r'] = np.where(np.isfinite(P.m1), P.m1 / 100, P.r_ytd)
chk = P[np.isfinite(P.m1) & np.isfinite(P.r_ytd)]
val_check = {'n': int(len(chk)), 'mad_pp': float(np.median(np.abs(chk.m1 / 100 - chk.r_ytd)) * 100), 'p99_absdiff_pp': float(np.quantile(np.abs(chk.m1 / 100 - chk.r_ytd), 0.99) * 100)}

# 基準：0050 含息月底；另用同一份報表內的「台灣50報酬指數」作敏感度
e = G.E50.resample('ME').last(); b50 = e.pct_change()
tw50 = P[P.name.str.contains('台灣50報酬指數|臺灣50報酬指數')].groupby('date').r.first()

# 改名串接：某名稱最後出現月 m-1、另一名稱首次出現月 m、成立日相同且唯一 → 同一檔
first = P.groupby('name').date.min(); last = P.groupby('name').date.max(); incn = P.groupby('name').inc.agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None)
link = {}
for nm, ld in last.items():
    if ld >= pd.Timestamp('2026-09-30') or incn.get(nm) is None: continue
    cand = [m for m in first.index if first[m] == ld + pd.offsets.MonthEnd(1) and incn.get(m) == incn[nm] and m != nm]
    if len(cand) == 1: link[nm] = cand[0]
def root(nm):
    # 往回找最早的名字當 fund_id
    inv = {v: k for k, v in link.items()}
    while nm in inv: nm = inv[nm]
    return nm
P['fund_id'] = P.name.map(root)
P = P.sort_values(['fund_id', 'date']).drop_duplicates(['fund_id', 'date'], keep='last')

active = P[~P.name.str.contains(EXCL) & ~P.cls.isin(['index_etf', 'china'])]
P[['date', 'fund_id', 'name', 'cls', 'inc', 'r']].to_parquet(G.DATA / 'fund_monthly.parquet')

LAST = pd.Timestamp('2026-09-30')
rows = []
for fid, g in active.groupby('fund_id'):
    s = g.set_index('date').r.dropna()
    s = s[s.index >= '2006-08-31']
    if len(s) < 24: continue
    cls = g.cls[g.cls != 'parent'].mode().iloc[0] if (g.cls != 'parent').any() else 'parent'
    bx = b50.reindex(s.index); x = (s - bx).dropna()
    xt = (s - tw50.reindex(s.index)).dropna()
    rec = {'fund_id': fid, 'last_name': g.name.iloc[-1], 'cls': cls, 'inc': incn.get(g.name.iloc[-1]), 'first': s.index.min(), 'last': s.index.max(),
           'n': len(x), 'alive': bool(g.date.max() >= LAST), 'IR': G.ir(x), 't': G.tstat(x), 'IR_vsTW50idx': G.ir(xt),
           'ex_ann_pp': float((np.prod(1 + s[x.index]) ** (12 / len(x)) - np.prod(1 + bx[x.index]) ** (12 / len(x))) * 100),
           'TE': float(x.std(ddof=1) * np.sqrt(12))}
    for nm, (a, b) in {'P0618': ('2006-01-01', '2018-12-31'), 'Y19': ('2019-01-01', '2026-12-31'), 'SEG1': ('2006-01-01', '2012-12-31'),
                       'SEG2': ('2013-01-01', '2018-12-31'), 'SEG3': ('2019-01-01', '2026-12-31')}.items():
        xx = x[(x.index >= a) & (x.index <= b)]
        rec[f'IR_{nm}'] = G.ir(xx) if len(xx) >= 24 else np.nan
        rec[f'n_{nm}'] = len(xx)
    rows.append(rec)
FI = pd.DataFrame(rows)
FI.to_parquet(G.DATA / 'fund_ir.parquet')

def q(s, a): s = s.dropna(); return float(s.quantile(a)) if len(s) else np.nan
def summ(df, col='IR'):
    s = df[col].dropna()
    return {'n': int(len(s)), 'p90_top10pct': q(s, 0.9), 'median': q(s, 0.5), 'p10': q(s, 0.1), 'max': float(s.max()) if len(s) else np.nan, 'mean': float(s.mean()) if len(s) else np.nan}

out = {'source': 'SITCA 月度基金績效評比（GitHub 封存 z-Wind/stockTools，commit 68ecb82，2026-11-02）；報表內公佈時間＝次月 5～10 日（PIT）',
       'months_parsed': int(sum(1 for v in meta.values() if v.get('n_part1_rows', 0) >= 100)), 'months_failed_old_layout': [k for k, v in meta.items() if v.get('n_part1_rows', 0) < 100],
       'return_validation_m1_vs_ytd_diff': val_check, 'n_name_links': len(link), 'tw50idx_vs_0050_monthly_corr': float(pd.concat([tw50, b50], axis=1).dropna().corr().iloc[0, 1])}
surv15 = FI[FI.alive & (FI.n >= 180)]
gen = surv15[surv15.cls == 'general']
out['survivors_ge15y_general'] = {**summ(gen), 'IR_P0618': summ(gen, 'IR_P0618'), 'IR_Y19': summ(gen, 'IR_Y19'), 'IR_vsTW50idx': summ(gen, 'IR_vsTW50idx'),
                                  'ex_ann_pp': summ(gen, 'ex_ann_pp'), 'TE': summ(gen, 'TE'), 'share_Y19_lt_P0618': float((gen.IR_Y19 < gen.IR_P0618).mean())}
out['survivors_ge15y_all_active_domestic'] = {**summ(surv15), 'by_cls': surv15.cls.value_counts().to_dict(), 'IR_P0618': summ(surv15, 'IR_P0618'), 'IR_Y19': summ(surv15, 'IR_Y19'),
                                              'share_Y19_lt_P0618': float((surv15.IR_Y19 < surv15.IR_P0618).mean()),
                                              'segments': {s: summ(surv15, f'IR_{s}') for s in ['SEG1', 'SEG2', 'SEG3']}}
# 存活者偏誤：2006-08 起就存在（或 2011-10 前已成立）而之後消失的同類基金
start_univ = FI[FI['first'] <= pd.Timestamp('2011-10-31')]
dead = start_univ[~start_univ.alive]
for lab, base in [('general', 'general'), ('all_active_domestic', None)]:
    S = surv15 if base is None else surv15[surv15.cls == base]
    Dd = dead if base is None else dead[dead.cls == base]
    lb = pd.concat([S.IR, pd.Series([S.IR.min()] * len(Dd))])
    act = pd.concat([S.IR, Dd.IR])
    out[f'survivorship_{lab}'] = {'n_survivors_ge15y': int(len(S)), 'n_dead_since_univ': int(len(Dd)), 'attrition_rate': float(len(Dd) / max(len(Dd) + len(start_univ[start_univ.alive & ((start_univ.cls == base) if base else True)]), 1)),
                                  'upper_bound_p90': q(S.IR, 0.9), 'upper_bound_median': q(S.IR, 0.5),
                                  'lower_bound_p90_registered_minimpute': q(lb, 0.9), 'lower_bound_median_registered_minimpute': q(lb, 0.5),
                                  'actual_partial_life_p90': q(act, 0.9), 'actual_partial_life_median': q(act, 0.5),
                                  'dead_IR_median': q(Dd.IR, 0.5), 'dead_n_median_months': float(Dd.n.median()) if len(Dd) else np.nan}
out['top_funds_general_ge15y'] = gen.sort_values('IR', ascending=False).head(12)[['last_name', 'IR', 't', 'IR_P0618', 'IR_Y19', 'ex_ann_pp', 'TE', 'n']].round(3).to_dict('records')
out['top_funds_all_ge15y'] = surv15.sort_values('IR', ascending=False).head(12)[['last_name', 'cls', 'IR', 't', 'IR_P0618', 'IR_Y19', 'ex_ann_pp', 'TE', 'n']].round(3).to_dict('records')
out['evidence_grade'] = 'A−（公開 SITCA 官方月報、本中心自行計算；但只經研究員一方計算；2006-01～07 缺；報酬為公會公布的含息報酬，未逐檔核對配息）'
ex = json.loads((G.HERE / 'explore.json').read_text()) if (G.HERE / 'explore.json').exists() else {}
ex['funds'] = out
(G.HERE / 'explore.json').write_text(json.dumps(ex, ensure_ascii=False, indent=1, default=str))
G.save_key('FUNDS', json.loads(json.dumps(out, default=str)))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:7000])
