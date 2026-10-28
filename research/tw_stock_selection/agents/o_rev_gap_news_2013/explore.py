#!/usr/bin/env python3
"""步驟 0／1 探索（不跑任何策略淨值）：
 (0) V4 淨值 2013～2021 跳日清點（o5 教訓 1）：相鄰兩筆淨值之間，加權指數交易日曆上缺幾個交易日；> 5 個交易日列出。
 (1) 缺口格清冊摘要：類型（併購／惡化／非下市）× 年、V60 排名組、52 週比組、B10 實際入選；非下市格逐格原因。
 (2) 機制描述（只用價格挑選期：訊號日 ≤ 2019-04-30）：缺口股個股下一窗報酬（訊號日收盤 → 下一訊號日收盤，自算還原價，
     停牌／下市後以最後收盤計）減同日 ok_other 池等權。依類型、52 週比、B10 入選分組。這是個股描述統計，不是策略報酬。
輸出 explore.json。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rglib as G
K, V, L = G.K, G.V, G.L
b = G.base(); S = G.mk()
X = pd.read_csv(HERE / 'gap_cells_2013_2021.csv', dtype={'code': str})
out = {}
# ---- (0) 跳日 ----
nv = pd.read_parquet(G.V4DIR / 'navs_v4.parquet')['AB_V4|normal'].dropna()
cal = pd.DatetimeIndex(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']).sort_values()
idx = nv.index
jumps = []
for a, c in zip(idx[:-1], idx[1:]):
    miss = int(((cal > a) & (cal < c)).sum())
    if miss > 0: jumps.append((str(a.date()), str(c.date()), miss))
out['step0_nav_jumps'] = {'rule': '相鄰淨值日之間，加權指數有交易但淨值沒有的交易日數',
                          'all_jumps_gt0': jumps, 'jumps_gt5': [j for j in jumps if j[2] > 5],
                          'in_2013_2021_gt5': [j for j in jumps if j[2] > 5 and '2013' <= j[0][:4] <= '2021'],
                          'calendar_gaps_ge9_days_2013_2021_note': '2013～2021 淨值相鄰間隔 ≥ 9 日曆日者全是農曆春節休市（加權指數同樣無交易）'}
# ---- (1) 清冊 ----
X['year'] = X.sig.str[:4]
X['liq_grp'] = pd.cut(X.liq_rank, [0, 100, 200, 300, 10 ** 6], labels=['1-100', '101-200', '201-300', '>300'])
X['s52_grp'] = pd.cut(X.s52_pct, [-0.01, 0.5, 0.7, 0.9, 1.0], labels=['<0.5', '0.5-0.7', '0.7-0.9', '>=0.9'])
inv = {'n_cells': int(len(X)), 'n_codes': int(X.code.nunique()), 'n_sig_days': int(X.sig.nunique()),
       'by_type': X.type.value_counts().to_dict(),
       'by_year_type': {y: g.type.value_counts().to_dict() for y, g in X.groupby('year')},
       'by_liq_grp': X.liq_grp.value_counts().sort_index().astype(int).to_dict(),
       'by_s52_grp': X.s52_grp.value_counts().sort_index().astype(int).to_dict(),
       'B10_picks': {'n': int(X.in_B10.sum()), 'by_type': X[X.in_B10].type.value_counts().to_dict(),
                     'by_code_top10': X[X.in_B10].code.value_counts().head(10).to_dict(),
                     'by_year': X[X.in_B10].year.value_counts().sort_index().to_dict()},
       'n_2013_2020': int((X.year <= '2020').sum()), 'n_nondelist_2013_2020': int(((X.year <= '2020') & (X.type == 'nondelist')).sum()),
       'by_code': X.groupby(['code', 'type']).size().sort_values(ascending=False).head(25).reset_index().astype(str).values.tolist()}
# 非下市格原因（逐格）
lr = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
first_lv = lr.groupby('code').month.min(); last_lv = lr.groupby('code').month.max()
r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str})
has5 = set(r5.code)
reasons = []
for r in X[X.type == 'nondelist'].itertuples():
    fl, ll = first_lv.get(r.code), last_lv.get(r.code)
    m = pd.Period(r.sig, 'M')
    if ll is not None and pd.Period(ll[:7], 'M') < m - 3 and r.code not in has5:
        why = '資料源漏抓（tw_revenue 該代碼 2013 起中斷、rev_5y 也無；公司仍上市）'
    elif fl is not None and pd.Period(fl[:7], 'M') > m - 16:
        why = '新代碼／控股公司轉換（首筆營收距訊號月 < 16 個月，缺年增率基期）'
    elif r.n_level_miss == 0 and r.n_base_miss >= 1:
        why = '基期月營收缺或 ≤ 0（年增率無定義）'
    else:
        why = '個別月份缺值'
    reasons.append((r.sig, r.code, r.name if isinstance(r.name, str) else '', why))
R = pd.DataFrame(reasons, columns=['sig', 'code', 'name', 'why'])
R.to_csv(HERE / 'data' / 'nondelist_reasons.csv', index=False)
inv['nondelist_reasons'] = R.groupby('why').size().to_dict()
inv['nondelist_reasons_by_code'] = R.groupby(['code', 'why']).size().reset_index().astype(str).values.tolist()
out['step1_inventory'] = inv
# ---- (2) 機制描述（挑選期）----
A = K.MYADJ.ffill()
sd = S.sig_days; nxt = {d: sd[i + 1] for i, d in enumerate(sd[:-1])}
rows = []
for d, g in X.groupby('sig'):
    d = pd.Timestamp(d)
    if d not in nxt or d > pd.Timestamp('2019-04-30'): continue
    cs = S.cross_section(d)
    r = (A.loc[nxt[d]] / A.loc[d] - 1).to_numpy()
    ew = np.nanmean(np.where(cs['ok_other'], r, np.nan))
    for c, p, t, bb in zip(g.code, g.s52_pct, g.type, g.in_B10):
        rows.append((d, c, p, t, bb, r[K.CI[c]] - ew))
E = pd.DataFrame(rows, columns=['d', 'code', 's52_pct', 'type', 'in_B10', 'ex'])
def bym(x):
    if len(x) == 0: return {'n_cells': 0}
    m = x.groupby('d').ex.mean()
    o = {'n_cells': int(len(x)), 'n_months': int(len(m)), 'mean_pp': float(m.mean() * 100),
         't_by_month': float(L.tstat(m.to_numpy())) if len(m) >= 3 else None}
    # 月份分群 bootstrap 95% 區間（事件少時必附）
    rng = np.random.default_rng(11); mv = m.to_numpy()
    bs = [rng.choice(mv, len(mv)).mean() for _ in range(5000)]
    o['boot95_pp'] = [float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)]
    return o
mech = {k: bym(v) for k, v in {
    'all': E, 'merger': E[E.type == 'merger'], 'deterior': E[E.type == 'deterior'], 'nondelist': E[E.type == 'nondelist'],
    's52_pct>=0.7': E[E.s52_pct >= 0.7], 's52_pct<0.7': E[E.s52_pct < 0.7],
    'merger_s52>=0.7': E[(E.type == 'merger') & (E.s52_pct >= 0.7)], 'deterior_s52>=0.7': E[(E.type == 'deterior') & (E.s52_pct >= 0.7)],
    'B10_picks': E[E.in_B10], 'B10_picks_merger': E[E.in_B10 & (E.type == 'merger')], 'B10_picks_deterior': E[E.in_B10 & (E.type == 'deterior')]}.items()}
out['step1_mechanism_descriptive（個股下一窗報酬減同池等權；挑選期 ≤ 2019-04；不是策略報酬）'] = mech
E.to_csv(HERE / 'data' / 'explore_gap_stock_excess.csv', index=False)
F = HERE / 'explore.json'
old = json.loads(F.read_text(encoding='utf-8')) if F.exists() else {}
old.update(out)
F.write_text(json.dumps(old, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({'jumps_gt5': out['step0_nav_jumps']['jumps_gt5'], 'inv': {k: inv[k] for k in ('n_cells', 'by_type', 'B10_picks', 'nondelist_reasons', 'n_2013_2020', 'n_nondelist_2013_2020')}, 'mech': mech}, ensure_ascii=False, indent=1, default=str))
