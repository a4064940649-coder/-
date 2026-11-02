#!/usr/bin/env python3
"""第 0 步：時間戳覆蓋閘門（不看任何報酬）。規則見 gate_prereg.json（先斷言登記存在且雜湊相符）。
來源：raw/screener（上游 tw-stock-screener blobless clone）、raw/tasi（shohokuno10/Tasi blobless clone，營收/YYYYMM.csv）。
輸出：gate.json、data/firstseen_hist.csv。
執行：python3 -I -B gate.py"""
import sys, json, re, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ts_common as T

G0 = T.check_gate_prereg()
access = []
def log(**k): k['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); access.append(k)

# ---------------- (b) Tasi ----------------
tasi = T.RAW / 'tasi'
files = [l for l in T.git(tasi, 'ls-tree', '--name-only', 'HEAD', '營收/').splitlines() if re.fullmatch(r'營收/\d{6}\.csv', l)]
vers_t = []; n_blob = 0; skipped_old = {}
for f in files:
    ym = f'{f[3:7]}-{f[7:9]}'
    cm = T.file_commits(tasi, f)
    if ym < '2024-04':
        # 2024-06 之前的營收月檔是建 repo 時一次加入的回填檔：首見＝檔案加入 commit（無 T_prev），依規則一律不算；不取 blob。
        skipped_old[ym] = dict(n_commits=len(cm), first=str(cm[0][1]) if cm else None)
        continue
    for h, t in cm:
        try:
            S = T.parse_tasi(T.blob(tasi, h, f), ym); n_blob += 1
        except Exception as e:  # 刪除或壞檔
            S = set()
        vers_t.append((f, h, t, S))
vers_t.sort(key=lambda r: r[2])
log(source='tasi', url='https://github.com/shohokuno10/Tasi', head=T.git(tasi, 'rev-parse', 'HEAD').strip(), n_files_rev=len(files),
    n_blobs_fetched=n_blob, old_files_not_fetched=len(skipped_old))
FS_T = T.first_seen(vers_t)

# ---------------- (a) 上游 tw-stock-screener ----------------
scr = T.RAW / 'screener'
UPF = {'data/extras/revenue.csv': T.parse_up_revenue_csv, 'data/extras/rev_hist.csv.gz': T.parse_up_long, 'data/extras/rev_5y.csv.gz': T.parse_up_long}
vers_u = []; ann = {}; n_blob = 0
for f, fn in UPF.items():
    for h, t in T.file_commits(scr, f):
        b = T.blob(scr, h, f); n_blob += 1
        try: S = fn(b)
        except Exception: S = set()
        vers_u.append((f, h, t, S))
        if f.endswith('rev_hist.csv.gz'):
            for k, v in T.parse_rev_hist_ann(b).items(): ann.setdefault(k, v)
vers_u.sort(key=lambda r: r[2])
log(source='upstream', url='https://github.com/stevenhsiehsheng-ctrl/tw-stock-screener', head=T.git(scr, 'rev-parse', 'HEAD').strip(),
    first_commit=T.git(scr, 'log', '--reverse', '--format=%cI').splitlines()[0], n_blobs_fetched=n_blob,
    files={f: len(T.file_commits(scr, f)) for f in UPF})
FS_U = T.first_seen(vers_u)

# GitHub 搜尋紀錄（未授權 API；只當 access_log）
sl = (T.RAW / 'search_log.txt')
log(source='github_search', queries=['taiwan revenue', 'monthly revenue taiwan', 'mops revenue', 'twse revenue', 'tw stock revenue', '台股 營收', '營收',
                                     't05st10', 'mops monthly revenue archive', 'twse revenue github actions', 'taiwan monthly revenue csv'],
    result='沒有 2019～2024-05 的逐次 commit 營收封存；找到的每日更新 repo（Kenmeibbo/taiwan-revenue-monitor、CBturbomax/taiwan-it-revenue、clhuang224/stock-revenue 等）'
           '都在 2026 年建立，只能落在 Tasi 已涵蓋的錨定月內，不可能改變閘門結論，未下載。o8 的 12＋4 組搜尋結論相同。',
    log_file=str(sl.relative_to(T.HERE)) if sl.exists() else None)

# ---------------- 分母 ----------------
E = T.engine(); OFF, K, STC = E['OFF'], E['K'], E['STC']
TCAL = STC.TCAL
sl_ = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str)
kydr = set(sl_.code[sl_.name.str.contains('KY', na=False)]) | {c for c in K.COLS if c.startswith('91')}
days = [d for d in OFF.sig_days if pd.Period('2019-05', 'M') <= pd.Period(d, 'M') <= pd.Period('2026-09', 'M')]
d_oct = TCAL[TCAL.searchsorted(pd.Timestamp('2026-10-11'))]
days.append(d_oct)
rows = []
for d in days:
    m = pd.Period(d, 'M'); cs = OFF.cross_section(d)
    pr = pd.Series(np.where(cs['el'], cs['y3'], np.nan)).rank(pct=True).to_numpy()
    for j in np.where(cs['ok_other'])[0]:
        c = K.COLS[j]
        rows.append((str(m), str(m - 1), c, str(d.date()), c in kydr, pr[j], bool(cs['el'][j])))
POP = pd.DataFrame(rows, columns=['anchor', 'rev_month', 'code', 'sig_day', 'kydr', 'rev_pct', 'el'])
n_kydr = int(POP.kydr.sum()); POP = POP[~POP.kydr].drop(columns='kydr').reset_index(drop=True)
o8 = pd.read_csv(T.AG / 'o_rev_freshness_attention' / 'data' / 'gate_population.csv.gz', dtype=str)
chk = dict(o8_n=len(o8), ours_le_2026_09=int((POP.anchor <= '2026-09').sum()),
           o8_overlap=len(set(zip(o8.anchor, o8.code)) & set(zip(POP.anchor, POP.code))))

# ---------------- 覆蓋 ----------------
def attach(FS, tag):
    v = [FS.get((c, y)) for c, y in zip(POP.code, POP.rev_month)]
    POP[f'{tag}_seen'] = [x is not None for x in v]
    POP[f'{tag}_valid'] = [bool(x and x['valid']) for x in v]
    POP[f'{tag}_reason'] = [x['reason'] if x else '' for x in v]
    POP[f'{tag}_T'] = [x['T'] if x else None for x in v]
    POP[f'{tag}_Tprev'] = [x['T_prev'] if x else None for x in v]
attach(FS_T, 'tasi'); attach(FS_U, 'up')

# ---------------- 交叉核對 ----------------
def interval_ok(a, b):
    lo = max(a[0], b[0]); hi = min(a[1], b[1]); return lo < hi
ov = POP[POP.tasi_valid & POP.up_valid]
agree_int = [interval_ok((r.tasi_Tprev, r.tasi_T), (r.up_Tprev, r.up_T)) for r in ov.itertuples()]
same_day = [T.announce_day(r.tasi_T, TCAL) == T.announce_day(r.up_T, TCAL) for r in ov.itertuples()]
def ann_agree(tag):
    sub = POP[POP[f'{tag}_valid']]
    res = []
    for r in sub.itertuples():
        a = ann.get((r.code, r.rev_month))
        if not a: continue
        ad = pd.Timestamp(a)
        lo = pd.Timestamp(getattr(r, f'{tag}_Tprev').astimezone(T.TPE).date()); hi = pd.Timestamp(getattr(r, f'{tag}_T').astimezone(T.TPE).date())
        res.append(lo <= ad <= hi)
    return dict(n=len(res), agree=float(np.mean(res)) if res else None)
cross = dict(tasi_vs_up=dict(n=len(ov), interval_agree=float(np.mean(agree_int)) if len(ov) else None,
                             same_announce_day=float(np.mean(same_day)) if len(ov) else None),
             tasi_vs_ann=ann_agree('tasi'), up_vs_ann=ann_agree('up'))
use_t = use_u = True
if cross['tasi_vs_up']['n'] and cross['tasi_vs_up']['interval_agree'] < 0.95:
    a_t = cross['tasi_vs_ann']['agree'] or 0; a_u = cross['up_vs_ann']['agree'] or 0
    if a_t < a_u: use_t = False
    else: use_u = False
# 既有首見表（只讀，描述用）
ext = {}
f8 = T.AG / 'o_rev_freshness_attention' / 'forward' / 'firstseen.csv'
if f8.exists():
    x = pd.read_csv(f8, dtype=str); ext['o8_forward_firstseen'] = dict(n=len(x), yms=sorted(x.ym.unique().tolist())[-4:], first_seen_min=x.first_seen.min())
fra = T.AG / 'rev_announce' / 'firstseen_2026-09.csv'
if fra.exists():
    x = pd.read_csv(fra, dtype=str)
    x.columns = ['code'] + list(x.columns[1:])
    m = POP[(POP.rev_month == '2026-09') & POP.up_valid]
    xx = dict(zip(x.code, x.first_seen))
    dif = []
    for r in m.itertuples():
        if r.code in xx and isinstance(xx[r.code], str):
            fs_ = pd.Timestamp(xx[r.code]).tz_convert(T.TPE) if pd.Timestamp(xx[r.code]).tzinfo else pd.Timestamp(xx[r.code]).tz_localize(T.TPE)
            dif.append((r.up_T.astimezone(T.TPE) - fs_.to_pydatetime()).total_seconds() / 3600)
    ext['rev_announce_firstseen_2026_09'] = dict(n_rows=len(x), n_overlap_up_valid=len(dif),
                                                 hours_up_minus_ra_median=float(np.median(dif)) if dif else None,
                                                 share_abs_le_24h=float(np.mean(np.abs(dif) <= 24)) if dif else None)
an5 = T.AG / 'o_rev_event_trigger' / 'data' / 'ann_news.csv'
if an5.exists():
    x = pd.read_csv(an5, dtype=str); ext['o5_ann_news'] = dict(n=len(x), rev_month_range=[x.rev_month.min(), x.rev_month.max()],
                                                               overlap_with_ts_months=int(x.rev_month.isin(POP.rev_month[POP.tasi_valid | POP.up_valid]).sum()))

POP['valid_any'] = (POP.tasi_valid & use_t) | (POP.up_valid & use_u)
POP['T_best'] = [min([t for t, ok in ((a, va and use_t), (b, vb and use_u)) if ok and t is not None], default=None)
                 for a, va, b, vb in zip(POP.tasi_T, POP.tasi_valid, POP.up_T, POP.up_valid)]
POP['Tprev_best'] = [ (a_p if (ta == tb) else b_p) if tb is not None else None for ta, tb, a_p, b_p in
                      zip(POP.tasi_T, POP.T_best, POP.tasi_Tprev, POP.up_Tprev)]
# 公告日與 rel
dl = {m: STC.deadline(pd.Period(m, 'M')) for m in POP.anchor.unique()}
def rel_of(r):
    if r.T_best is None or pd.isna(r.T_best): return None, None
    a = T.announce_day(r.T_best, TCAL); D = dl[r.anchor]
    if a is None or D is None: return a, None
    return a, int(TCAL.get_loc(D) - TCAL.get_loc(a))
ar = [rel_of(r) for r in POP.itertuples()]
POP['a_day'] = [str(a.date()) if a is not None else '' for a, _ in ar]
POP['rel'] = [x for _, x in ar]
POP['deadline'] = [str(dl[m].date()) if dl[m] is not None else '' for m in POP.anchor]

cov = lambda s: float(s.mean())
G = dict(prereg=dict(script_sha256=G0['script_sha256'], rules_sha256=G0['rules_sha256'], utc=G0['utc']),
         denominator=dict(n_stock_months=len(POP), n_anchors=int(POP.anchor.nunique()), anchors=[POP.anchor.min(), POP.anchor.max()],
                          kydr_removed=n_kydr, o8_check=chk, sig_day_2026_10=str(d_oct.date())),
         coverage=dict(tasi_seen=cov(POP.tasi_seen), tasi_valid=cov(POP.tasi_valid), up_seen=cov(POP.up_seen), up_valid=cov(POP.up_valid),
                       union_valid=cov(POP.valid_any), threshold=0.60),
         reasons=dict(tasi=POP[POP.tasi_seen].tasi_reason.value_counts().to_dict(), up=POP[POP.up_seen].up_reason.value_counts().to_dict()),
         by_year=POP.assign(y=POP.anchor.str[:4]).groupby('y')[['tasi_valid', 'up_valid', 'valid_any']].mean().round(4).to_dict('index'),
         by_month={k: v for k, v in POP.groupby('anchor')[['tasi_seen', 'tasi_valid', 'up_valid', 'valid_any']].mean().round(4).to_dict('index').items()
                   if k >= '2024-06'},
         upper_bound=dict(tasi_window_anchor_share=float((POP.anchor >= '2024-06').mean()),
                          note='即使 Tasi 期間（錨定 2024-06～2026-10）每個股月都有合格時間戳，覆蓋也只有這個比例；2019-05～2024-05 沒有任何逐次 commit 來源。'),
         cross_check=cross, sources_counted=dict(tasi=use_t, upstream=use_u), external_firstseen=ext,
         tasi_old_files_skipped=len(skipped_old), access_log=access)
G['pass'] = bool(G['coverage']['union_valid'] >= 0.60)
G['branch'] = 'B' if G['pass'] else 'A'
G['verdict'] = (f"合計覆蓋 {G['coverage']['union_valid']:.1%}（< 60%）→ 分支 A：不做任何歷史報酬檢定，只做不看報酬的描述統計與前瞻紀錄。"
                if not G['pass'] else '合計覆蓋 ≥ 60% → 分支 B')
T.jdump(G, T.HERE / 'gate.json')
out = POP.copy()
for c in ['tasi_T', 'tasi_Tprev', 'up_T', 'up_Tprev', 'T_best', 'Tprev_best']:
    out[c] = [x.astimezone(T.TPE).isoformat() if (x is not None and not pd.isna(x)) else '' for x in out[c]]
out.to_csv(T.DATA / 'firstseen_hist.csv', index=False)
# 來源層級的全部首見（不限股池），供前瞻與複製
allfs = [dict(source=s, code=c, ym=y, T=v['T'].astimezone(T.TPE).isoformat(), T_prev=v['T_prev'].astimezone(T.TPE).isoformat() if v['T_prev'] else '',
              file=v['file'], commit=v['commit'], valid=v['valid'], reason=v['reason'])
         for s, FS in (('tasi', FS_T), ('upstream', FS_U)) for (c, y), v in FS.items() if y >= '2024-04']
pd.DataFrame(allfs).to_csv(T.DATA / 'firstseen_sources.csv.gz', index=False)
print(json.dumps(T.conv({k: G[k] for k in ('denominator', 'coverage', 'reasons', 'cross_check', 'sources_counted', 'upper_bound', 'verdict')}), ensure_ascii=False, indent=1))
