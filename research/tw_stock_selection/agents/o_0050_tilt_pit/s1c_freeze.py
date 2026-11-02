#!/usr/bin/env python3
"""凍結 PIT 股數（看報酬之前）＋重建 2014-01～2015-09 的臺灣50 成分（用 PIT 股數；規則同 o7：審核基準日市值排名，非成分 ≤ 40 納入、成分 > 60 剔除、補足 50）。
2015-09-18 以後用公開名單（tw50_members.json）。種子＝2013-12 審核基準日 PIT 市值前 50；與 2015-09-18 生效前的實際名單比對一致率。
寫 data/membership_pit.json、results.json['gate']（含 sha256、時間戳、可回測段落）。"""
import sys, json, math, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tlib as L
RAW, DATA = L.RAW, L.DATA
S = pd.read_parquet(DATA / 'shares_pit.parquet'); ID = S.index; COLS = S.columns; CI = L.CI
CFF = pd.DataFrame(L.CFF, index=L.IDX, columns=L.COLS).reindex(ID)
DR = np.array([c.startswith('91') for c in COLS])
TW = json.loads((RAW / 'agents' / 'tw50_members.json').read_text(encoding='utf-8'))
EARLIEST = pd.Timestamp(TW['earliest_valid_eff'])
def third_friday(y, m):
    d = pd.Timestamp(y, m, 15); return d + pd.Timedelta(days=(4 - d.weekday()) % 7)
evc = pd.read_csv(RAW / 'agents' / 'etf_rebalance_events.csv', dtype=str); evc = evc[evc.etf == 'TW50']
CUT = {r.review: (pd.Timestamp(r.eff), pd.Timestamp(r.cutoff)) for r in evc.itertuples()}
TR = L.C.reindex(ID).notna().to_numpy()
def mcap_at(t):
    k = ID.get_loc(t)
    mc = (S.iloc[k] * CFF.iloc[k]).to_numpy()
    alive = TR[max(k - 5, 0):k + 1].any(axis=0)
    return np.where(~DR & np.isfinite(mc) & alive, mc, -np.inf)
def rank_rule(prev, t):
    mc = mcap_at(t); order = np.argsort(-mc); rank = np.empty(len(COLS), int); rank[order] = np.arange(1, len(COLS) + 1)
    keep = [c for c in prev if c in CI and rank[CI[c]] <= 60 and np.isfinite(mc[CI[c]])]
    add_ = [COLS[j] for j in order[:40] if COLS[j] not in prev]
    S_ = list(dict.fromkeys(keep + add_))
    if len(S_) > 50: S_ = sorted(S_, key=lambda c: rank[CI[c]])[:50]
    k = 0
    while len(S_) < 50:
        c = COLS[order[k]]; k += 1
        if c not in S_: S_.append(c)
    return S_
memb = {}
cut0 = ID[ID <= third_friday(2013, 12) - pd.Timedelta(days=25)][-1]
mc0 = mcap_at(cut0); cur = [COLS[j] for j in np.argsort(-mc0)[:50]]
memb[str(third_friday(2013, 12).date())] = cur
for y in (2014, 2015):
    for m in (3, 6, 9, 12):
        key = f'{y}-{m:02d}'
        eff, cut = CUT.get(key, (third_friday(y, m), third_friday(y, m) - pd.Timedelta(days=25)))
        if eff >= EARLIEST: continue
        cur = rank_rule(cur, ID[ID <= cut][-1]); memb[str(eff.date())] = cur
actual_0915 = None
for s in TW['steps_backward']:
    if pd.Timestamp(s['eff']) == EARLIEST: actual_0915 = s['members_before']
agree = len(set(cur) & set(actual_0915)) / 50 if actual_0915 else None
(DATA / 'membership_pit.json').write_text(json.dumps(memb, ensure_ascii=False, indent=1), encoding='utf-8')
G = json.loads((DATA / 'gate.json').read_text(encoding='utf-8'))
res_f = L.HERE / 'results.json'
R = json.loads(res_f.read_text(encoding='utf-8')) if res_f.exists() else {}
R['gate'] = {
    'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'shares_pit_sha256': L.sha(DATA / 'shares_pit.parquet'), 'shares_src_sha256': L.sha(DATA / 'shares_src.parquet'),
    'membership_pit_sha256': L.sha(DATA / 'membership_pit.json'),
    'checkpoints': {k: {x: v[x] for x in ['date', 'independent_truth', 'n_top100_truth', 'coverage', 'median_abs_err', 'p90_abs_err', 'share_err_gt_5pct', 'spearman_mcap_rank', 'pass', 'worst', 'excluded_same_truth_anchor']}
                    for k, v in G.items() if isinstance(v, dict) and 'median_abs_err' in v},
    'coverage_top100': G['coverage_top100'], 'identical_2019_06_vs_2021_10': G['identical_2019_06_vs_2021_10'],
    'delisted_ever_top120_2013_2026': G['delisted_ever_top120_2013_2026'],
    'membership_rule_2014_2015': {'reviews': list(memb), 'agree_with_actual_before_2015-09-18': agree},
    'segments': {'2006-04～2013-12': '不過閘門（2006、2010 檢核點無公開真值；2013 年無除權事件資料，金融股 2015Q1 前只能用 2014 起的 TWT49U 回推）→ 不回測',
                 '2014-01～2026-10': '通過（2015、2017、2021×2 獨立檢核點中位誤差 ≤ 0.05%、Spearman ≥ 0.998；前 100 名 PIT 覆蓋 100%；2019-06/2021-10 股數相同比例 44% 對實際 49%）'},
    'selection_period_effective': '2014-01～2018-12（原計畫 2006-04～2018-12，前段不過閘門）',
}
res_f.write_text(L.dumps(R), encoding='utf-8')
print('membership agree 2015-09:', agree, ' reviews:', list(memb))
print(json.dumps(R['gate']['segments'], ensure_ascii=False, indent=1))
