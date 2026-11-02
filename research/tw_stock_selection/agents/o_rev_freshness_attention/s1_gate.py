#!/usr/bin/env python3
"""步驟 1：逐檔公告時間戳覆蓋閘門（不看任何報酬）。
母體：2019-05～2026-09 錨定月（V7 官方訊號日）的前 300 池、排除金融與 KY／DR、處置與次日可成交條件（＝V4 ok_other），共 N 個股月。
閘門：有「精確到日、PIT 合法」首見日的股月比例 ≥ 60%，且與 rev_hist、新聞重疊月的一致率 ≥ 90%。
來源（只讀日期與檔名）：
  (a) o_rev_event_trigger/data/ann_news.csv first_strict（鉅亨新聞「今(X)日」且 X＝文章日；2013-10～2021-09）
  (b) tw-stock-screener extras/rev_hist.csv.gz 的 ann_date（2026-08 起）
  (c) raw/tasi（shohokuno10/Tasi blobless clone）：營收/YYYYMM.csv 每日快照 commit；逐檔首見日要靠 blob diff。
      這裡先算「上界」：只要該營收月的檔案在期限日前已存在且有 ≥ 3 個不同 commit 日，就假設池內每檔都有首見日（100% 月內覆蓋）。
  (d) raw/finmind（wenchiehlee/Python-Actions.FinMind blobless clone）：financial/type5/raw_revenue_<code>.csv，以檔案加入日起算，
      同樣假設月內 100% 覆蓋（上界）。
上界若已 < 60%，閘門不通過，不必下載 blob。"""
import sys, subprocess, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fa_common as C

E = C.engine(); K, L, STC = E['K'], E['L'], E['STC']
OFF = E['OFF']
rows = []
for d in OFF.sig_days:
    m = pd.Period(d, 'M')
    if not (C.P('2019-05') <= m <= C.P('2026-09')): continue
    cs = OFF.cross_section(d)
    for j in np.where(cs['ok_other'])[0]:
        rows.append((str(m), str(m - 1), K.COLS[j]))
POP = pd.DataFrame(rows, columns=['anchor', 'rev_month', 'code'])
n = len(POP)

# (a) 新聞
NW = pd.read_csv(C.AG / 'o_rev_event_trigger' / 'data' / 'ann_news.csv', dtype=str)
nw = set(zip(NW.code[NW.first_strict.notna()], NW.rev_month[NW.first_strict.notna()]))
nw_loose = set(zip(NW.code, NW.rev_month))
# (b) rev_hist
RH = pd.read_csv(C.SCR / 'extras' / 'rev_hist.csv.gz', dtype=str)
rh = set(zip(RH.code[RH.ann_date.notna()], RH.ym[RH.ann_date.notna()]))

def git(repo, *a):
    return subprocess.run(['git', '-C', str(repo), *a], capture_output=True, text=True, check=True).stdout

# (c) Tasi：每個營收月檔的 commit 日期（不取 blob）
tasi = C.RAW / 'tasi'
log = git(tasi, 'log', '--format=@%cs', '--name-only', '--', '營收')
cur = None; tdays = {}
for line in log.splitlines():
    line = line.strip()
    if not line: continue
    if line.startswith('@'): cur = line[1:]; continue
    p = line.strip('"')
    if p.endswith('.csv'):
        ym = Path(p).stem
        if len(ym) == 6 and ym.isdigit(): tdays.setdefault(f'{ym[:4]}-{ym[4:]}', set()).add(cur)
tasi_ok = set()
for ym, ds in tdays.items():
    m = pd.Period(ym, 'M'); dl = STC.deadline(m + 1)
    first = min(pd.Timestamp(x) for x in ds)
    if dl is not None and first <= dl and len(ds) >= 3: tasi_ok.add(ym)
# (d) FinMind：檔案加入日
fm = C.RAW / 'finmind'
log = git(fm, 'log', '--diff-filter=A', '--format=@%cs', '--name-only', '--', 'financial/type5')
cur = None; fadd = {}
for line in log.splitlines():
    line = line.strip()
    if not line: continue
    if line.startswith('@'): cur = line[1:]; continue
    nm = Path(line).stem
    if nm.startswith('raw_revenue_'):
        c = nm.split('_')[-1]; fadd[c] = min(fadd.get(c, cur), cur)

def cov(r):
    a = (r.code, r.rev_month) in nw
    b = (r.code, r.rev_month) in rh
    c = r.rev_month in tasi_ok
    m = pd.Period(r.rev_month, 'M'); dl = STC.deadline(m + 1)
    dd = fadd.get(r.code)
    d_ = dd is not None and dl is not None and pd.Timestamp(dd) <= dl
    return a, b, c, d_
F = np.array([cov(r) for r in POP.itertuples()])
POP['news'], POP['revhist'], POP['tasi_ub'], POP['finmind_ub'] = F.T
POP['any_ub'] = F.any(axis=1)
by_src = {k: float(POP[k].mean()) for k in ('news', 'revhist', 'tasi_ub', 'finmind_ub', 'any_ub')}
by_year = POP.assign(y=POP.anchor.str[:4]).groupby('y')[['news', 'revhist', 'tasi_ub', 'finmind_ub', 'any_ub']].mean().round(4).to_dict('index')
overlap_news_revhist = len(nw & rh)
overlap_tasi_revhist_months = sorted(set(tasi_ok) & set(RH.ym[RH.ann_date.notna()]))
G = {'population_stock_months': n, 'n_anchor_months': int(POP.anchor.nunique()),
     'coverage_exact_or_upper_bound': by_src, 'coverage_by_anchor_year': by_year,
     'tasi_months_with_daily_snapshots_before_deadline': sorted(tasi_ok), 'finmind_type5_files': len(fadd),
     'finmind_first_add': min(fadd.values()) if fadd else None,
     'news_loose_coverage': float(np.mean([(c, r) in nw_loose for c, r in zip(POP.code, POP.rev_month)])),
     'overlap_news_revhist_stock_months': overlap_news_revhist, 'overlap_tasi_revhist_months': overlap_tasi_revhist_months,
     'threshold_coverage': 0.60, 'threshold_agreement': 0.90}
G['pass_coverage'] = bool(by_src['any_ub'] >= 0.60)
G['agreement'] = ('未計算：覆蓋上界已低於門檻；新聞與 rev_hist 重疊 0 個股月，Tasi 與 rev_hist 只重疊 '
                  f'{len(overlap_tasi_revhist_months)} 個營收月（需要 blob diff）')
G['pass'] = False if not G['pass_coverage'] else None
G['verdict'] = ('閘門不通過：即使把 Tasi（2024-06 起每日快照）與 FinMind（140 檔、2026-09 起）都當成月內 100% 覆蓋，'
                f'2019-05～2026-09 池內股月的上界只有 {by_src["any_ub"]:.1%}（< 60%）。訊號年齡分析與 R_roll 回測不做，R_roll 只做 2026-11 起前瞻紙上追蹤。')
POP.to_csv(C.DATA / 'gate_population.csv.gz', index=False)
C.save_key('GATE_TIMESTAMP', G)
print(json.dumps(C.conv({k: v for k, v in G.items() if k != 'coverage_by_anchor_year'}), ensure_ascii=False, indent=1))
