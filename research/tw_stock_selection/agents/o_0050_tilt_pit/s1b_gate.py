#!/usr/bin/env python3
"""PIT 股數閘門（凍結前、不看報酬）。檢核點：2015-05-22、2017-07-06（weihautin 全市場股本快照）、2019-06（jhc 交叉來源）、
2021-07-02／2021-09-24（TDCC 集保合計股數，獨立來源）、2025-10（MI_QFIIS，建構來源，只列參考）；2006、2010 無可核對公開真值。
閘門：市值前 100 名股數中位絕對誤差 ≤ 2%、市值名次 Spearman ≥ 0.995；前 100 名股月 PIT 覆蓋 ≥ 95%（含下市股）；
2019-06 與 2021-10 股數完全相同比例須接近實際股本不變比例（不能出現 o7 的 96.3%）。寫 data/gate.json 與 results.json['gate']。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import tlib as L
RAW, DATA = L.RAW, L.DATA
S = pd.read_parquet(DATA / 'shares_pit.parquet'); SRC = pd.read_parquet(DATA / 'shares_src.parquet')
ID = S.index; COLS = S.columns
CFF = pd.DataFrame(L.CFF, index=L.IDX, columns=L.COLS).reindex(ID)
SE = pd.read_parquet(RAW / 'agents' / 'capmatched_shares_est.parquet'); SE.index = pd.to_datetime(SE.index); SE = SE.reindex(index=ID, columns=COLS)
DR = np.array([c.startswith('91') for c in COLS])
G = {}
B1 = json.loads((DATA / 's1_build.json').read_text(encoding='utf-8'))
SNAPC = {k: set(v) for k, v in B1.get('snapshot_anchor_codes', {}).items()}
EXCL = {'2015-05-22_weihautin': SNAPC.get('weihautin_2015-05-22', set()), '2017-07-06_weihautin': SNAPC.get('weihautin_2017-07-06', set()),
        '2021-07-02_TDCC': SNAPC.get('TDCC_2021-07-02', set()), '2021-09-24_TDCC': SNAPC.get('TDCC_2021-09-24', set())}
def tday(d): return ID[ID <= pd.Timestamp(d)][-1]
def checkpoint(name, d, truth, independent=True):
    t = tday(d); px = CFF.loc[t]
    tr = truth[truth.index.isin(COLS) & (truth > 0)]
    tr = tr[[not c.startswith('91') for c in tr.index]]
    mc_true = (tr * px.reindex(tr.index)).dropna().sort_values(ascending=False)
    top_all = mc_true.index[:100]
    ex = EXCL.get(name, set())
    top = pd.Index([c for c in top_all if c not in ex])
    est = S.loc[t, top]
    e = np.log(est / tr[top]).dropna()
    cov = float(est.notna().mean())
    mc_est = (est * px[top])
    ok = mc_est.notna()
    rho = spearmanr(mc_true[top][ok], mc_est[ok]).correlation if ok.sum() > 10 else float('nan')
    worst = e.abs().sort_values(ascending=False).head(8)
    G[name] = {'date': str(t.date()), 'independent_truth': independent, 'n_top100_truth': int(len(top)), 'excluded_same_truth_anchor': sorted(set(top_all) & ex), 'coverage': cov,
               'median_abs_err': float(e.abs().median()), 'p90_abs_err': float(e.abs().quantile(0.9)), 'share_err_gt_5pct': float((e.abs() > 0.05).mean()),
               'spearman_mcap_rank': float(rho), 'pass': bool(e.abs().median() <= 0.02 and rho >= 0.995 and cov >= 0.95),
               'worst': {c: round(float(np.exp(e[c]) - 1), 4) for c in worst.index},
               'missing': [c for c in top if pd.isna(est[c])]}
# weihautin 2015-05-22（股本億，面額視為 10，已知面額變更的只有 2327/6415 在 2022 後）
w15 = pd.read_csv(RAW / 'weihautin' / 'update_csv' / 'StockList_20150522.csv', dtype={'代號': str})
checkpoint('2015-05-22_weihautin', '2015-05-22', w15.set_index('代號')['股本(億)'].astype(float) * 1e8 / 10)
w17 = pd.read_csv(RAW / 'weihautin' / 'update_csv' / 'StockList.csv', dtype={'代號': str})
w17 = w17.set_index('代號'); checkpoint('2017-07-06_weihautin', '2017-07-06', w17['股本(億)'].astype(float) * 1e8 / w17['面值(元)'].astype(float))
# jhc 2019Q2（OrdinaryShare，公告日 2019-08-xx）對建構值 2019-06-28（建構優先用 hawan/twsec；jhc 只補缺）
jh = {}
for f in (RAW / 'jhc' / 'GAT-main' / 'data').glob('*.csv'):
    d = pd.read_csv(f, usecols=['date', 'OrdinaryShare']); d = d[d.date.astype(str).str.startswith('2019-08')]
    if len(d): jh[f.stem] = float(d.OrdinaryShare.iloc[0]) / 10
checkpoint('2019-06-28_jhc_crosssource', '2019-06-28', pd.Series(jh), independent=False)
for f in ['2021-07-02', '2021-09-24']:
    td = pd.read_csv(RAW / 'tdcc' / f'total_{f}.csv', dtype={'code': str}).set_index('code')['shares'].astype(float)
    checkpoint(f'{f}_TDCC', f, td)
vf = pd.read_parquet(RAW / 'agents' / 'vf_twse_foreign.parquet'); vf = vf[vf.date.astype(str) == '2025-10-07'].set_index('code')['total_shares'].astype(float)
checkpoint('2025-10-07_MIQFIIS_construction', '2025-10-07', vf, independent=False)

# ---------------- 覆蓋：每月換股日的市值前 100 名（缺 PIT 者用 o7 回推估市值只為了決定名次）----------------
days = L.official_days('2013-06-01')
rows = []; delist_top = {}
PIT_CODES = {1, 2, 3, 4}
for d in days:
    t = tday(d)
    s = S.loc[t]; s_fb = s.fillna(SE.loc[t])
    mc = (s_fb * CFF.loc[t]).where(~DR & L.C.reindex(ID).loc[ID[(ID <= t)][-6:]].notna().any().to_numpy())
    top = mc.dropna().sort_values(ascending=False).index[:120]
    src = SRC.loc[t, top]
    for k, c in enumerate(top):
        pit = (int(src[c]) in PIT_CODES) and pd.notna(s[c])
        rows.append((d, c, k + 1, pit, int(src[c]) if pd.notna(s[c]) else -1))
        if L.LAST_TRADE[L.CI[c]] < len(L.IDX) - 1:
            delist_top.setdefault(c, {'last_trade': str(L.IDX[L.LAST_TRADE[L.CI[c]]].date()), 'best_rank': k + 1, 'months_top120': 0, 'pit_months': 0})
            delist_top[c]['best_rank'] = min(delist_top[c]['best_rank'], k + 1); delist_top[c]['months_top120'] += 1; delist_top[c]['pit_months'] += int(pit)
R = pd.DataFrame(rows, columns=['d', 'code', 'rank', 'pit', 'src'])
R['y'] = pd.DatetimeIndex(R.d).year
t100 = R[R['rank'] <= 100]
cov_y = t100.groupby('y').pit.mean().round(4).to_dict()
G['coverage_top100'] = {'overall': float(t100.pit.mean()), 'by_year': cov_y, 'src_mix': t100.src.value_counts().to_dict(),
                        'non_pit_examples': t100[~t100.pit].groupby('code').size().sort_values(ascending=False).head(15).to_dict()}
G['delisted_ever_top120_2013_2026'] = delist_top
# ---------------- 2019-06 vs 2021-10 股數完全相同比例 ----------------
t1, t2 = tday('2019-06-28'), tday('2021-10-29')
top19 = R[(R.d == [x for x in days if x.year == 2019 and x.month == 6][0]) & (R['rank'] <= 100)].code
a, b = S.loc[t1, top19], S.loc[t2, top19]; ok = a.notna() & b.notna()
same_est = float((np.abs(np.log(b[ok] / a[ok])) < 1e-9).mean())
H = pd.read_csv(RAW / 'hawan' / 'capital_rows.csv', dtype={'stock_id': str}); H = H[H.type == 'OrdinaryShare']
hp = H.pivot_table(index='date', columns='stock_id', values='value')
hp.index = pd.to_datetime(hp.index)
da_ = hp.index[np.argmin(np.abs((hp.index - pd.Timestamp('2019-06-30')).days))]; db_ = hp.index[np.argmin(np.abs((hp.index - pd.Timestamp('2021-09-30')).days))]
hh = hp.loc[[da_, db_]].T.dropna(); hh = hh[hh.index.isin(top19)]
same_true = float((np.abs(np.log(hh[db_] / hh[da_])) < 1e-9).mean())
SEa, SEb = SE.loc[t1, top19], SE.loc[t2, top19]; ok2 = SEa.notna() & SEb.notna()
G['identical_2019_06_vs_2021_10'] = {'pit_est': same_est, 'n': int(ok.sum()), 'statement_truth_hawan': same_true, 'n_truth': int(len(hh)), 'truth_dates': [str(da_.date()), str(db_.date())],
                                     'o7_shares_est': float((np.abs(np.log(SEb[ok2] / SEa[ok2])) < 1e-9).mean())}
G['identical_2019_06_vs_2021_10']['pass'] = bool(abs(same_est - same_true) <= 0.10)
# ---------------- 閘門判定 ----------------
cps = [k for k in G if k.endswith(('_weihautin', '_TDCC'))]
G['gate'] = {'checkpoints_independent': {k: G[k]['pass'] for k in cps},
             'coverage_pass': bool(G['coverage_top100']['overall'] >= 0.95),
             'identical_pass': G['identical_2019_06_vs_2021_10']['pass'],
             'pre2013_2006_2010': '不過（無可核對真值、無事件資料）→ 2006～2013-05 不回測',
             'segments': {}}
(DATA / 'gate.json').write_text(L.dumps(G), encoding='utf-8')
print(L.dumps({k: (v if k not in ('delisted_ever_top120_2013_2026',) else {c: x for c, x in list(v.items())[:40]}) for k, v in G.items()})[:9000])
