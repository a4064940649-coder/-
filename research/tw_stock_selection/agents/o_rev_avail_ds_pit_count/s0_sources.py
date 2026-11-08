#!/usr/bin/env python3
"""第 0 步＋探索（只看覆蓋與時間戳一致性，不看名次變化、不看報酬、不算任何 D=S 槽位的可得性比例）。
- 磁碟檢查（≥ 1.5GB）；連線測試結果（raw/step0_connectivity.txt，已在 2026-11-07T23:05Z 各測一次，不重試）。
- 把時間戳來源複製進 raw/（視為不可信、只當 CSV 讀）：
  ts_tasi_upstream_firstseen.csv.gz（o11 由 Tasi 與上游 git 歷史整理的首見表）、ts_mops_ann_date.csv（rev_hist ann_date）、
  ts_news_mentions.csv.gz（o6 由鉅亨／PTT／ELD／BOF 解析的營收提及）、tasi_commit_times.txt（Tasi 快照時刻）、
  disposal_official_universe.csv、disposal_hist_rejudge.csv。
- 覆蓋率：四段（2006～2013、2014～2018、2019～2024-05、2024-06 起）× 各來源，分母＝D=S／D>S 訊號日的 C0 股池股月（不是槽位）。
- 一致性閘門：Tasi 對 MOPS ann_date、Tasi 對上游、新聞數值對真值。
輸出 explore.json。python3 -I -B s0_sources.py"""
import sys, json, shutil, subprocess, gzip, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
AG = HERE.parent
RAW = HERE / 'raw'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o17_revavail')
import hashlib
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
T0 = time.time()
st = shutil.disk_usage('/')
DISK = {'free_GB': round(st.free / 1e9, 3), 'ok_ge_1p5GB': st.free / 1e9 >= 1.5}
assert DISK['ok_ge_1p5GB'], DISK

# ---------- 複製 raw ----------
man = {}
def put(src, dst, note):
    shutil.copyfile(src, RAW / dst); man[dst] = {'from': str(src), 'sha256': sha(RAW / dst), 'note': note}
put(AG / 'o_rev_event_clock_ts/data/firstseen_sources.csv.gz', 'ts_tasi_upstream_firstseen.csv.gz',
    'o11：shohokuno10/Tasi「營收/YYYYMM.csv」每日快照（2024-06-14 起）與上游 tw-stock-screener git（2026-09-29 起）的逐檔首見 T、前一張快照 T_prev')
rh = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_hist.csv.gz', dtype=str)
rh = rh[rh.ann_date.notna() & (rh.ann_date != '')][['code', 'ym', 'ann_date']]
rh.to_csv(RAW / 'ts_mops_ann_date.csv', index=False)
man['ts_mops_ann_date.csv'] = {'from': 'tw-stock-screener/data/extras/rev_hist.csv.gz（ann_date＝MOPS 公告日欄位，只有日期）', 'sha256': sha(RAW / 'ts_mops_ann_date.csv'), 'n': int(len(rh))}
mp = pd.read_parquet(AG / 'o_rev_gap_news_2013/data/mentions_P1.parquet')
mp = mp[['src', 'art', 'ts', 'has_time', 'code', 'ym', 'rev_k', 'yoy', 'mom', 'snip']].copy()
mp['snip'] = mp['snip'].astype(str).str.slice(0, 90)
mp['ts'] = mp['ts'].astype(str)
mp.to_csv(RAW / 'ts_news_mentions.csv.gz', index=False, compression='gzip')
man['ts_news_mentions.csv.gz'] = {'from': 'o_rev_gap_news_2013/data/mentions_P1.parquet（鉅亨 CTBC 鏡像 2013-01～2021-12 只有日期；PTT 2019～2021 檔名日期；ELD 2016～2018 含時間；BOF 檔名日期）',
                                  'sha256': sha(RAW / 'ts_news_mentions.csv.gz'), 'n': int(len(mp))}
lg = subprocess.run(['git', '-C', '/tmp/tasi_hist', 'log', '--format=%H %cI'], capture_output=True, text=True).stdout
(RAW / 'tasi_commit_times.txt').write_text(lg, encoding='utf-8')
man['tasi_commit_times.txt'] = {'from': '/tmp/tasi_hist（github.com/shohokuno10/Tasi blobless clone，HEAD 25fa2103 2026-10-27）git log', 'sha256': sha(RAW / 'tasi_commit_times.txt'),
                                'n_commits': len(lg.strip().splitlines())}
put(AG / 'o_disposal_official_fullhist/universe_flags.csv', 'disposal_official_universe.csv', '官方全歷史處置（普通股）事件期 start～end（o_disposal_official_fullhist）')
put(AG / 'o_disposal_rejudge/disposal_hist_rejudge.csv', 'disposal_hist_rejudge.csv', '官方處置公告全歷史（含公告日、措施）')
(RAW / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')

# ---------- 覆蓋 ----------
DD = pd.read_csv(HERE / 'data/census_days.csv', dtype=str)
PL = pd.read_csv(SCR / 'c0_pool.csv.gz', dtype={'code': str, 'sig': str})
PL = PL.merge(DD[['sig', 'case', 'rev_month', 'seg4']], on='sig')
FS = pd.read_csv(RAW / 'ts_tasi_upstream_firstseen.csv.gz', dtype=str)
NW = pd.read_csv(RAW / 'ts_news_mentions.csv.gz', dtype={'code': str, 'ym': str, 'art': str})
AN = pd.read_csv(RAW / 'ts_mops_ann_date.csv', dtype=str)
keyT = set(zip(FS[FS.source == 'tasi'].code, FS[FS.source == 'tasi'].ym))
keyU = set(zip(FS[FS.source == 'upstream'].code, FS[FS.source == 'upstream'].ym))
keyN = set(zip(NW.code, NW.ym))
keyA = set(zip(AN.code, AN.ym))
X = PL[PL.case.isin(['D=S', 'D>S'])].copy()
k = list(zip(X.code, X.rev_month))
for nm, ks in (('mops_ann', keyA), ('tasi', keyT), ('upstream', keyU), ('news', keyN)):
    X[nm] = [z in ks for z in k]
X['any'] = X[['mops_ann', 'tasi', 'upstream', 'news']].any(axis=1)
cov = {s: {c: round(float(g[c].mean()), 4) for c in ('mops_ann', 'tasi', 'upstream', 'news', 'any')} | {'n_stockmonths': int(len(g)), 'n_days': int(g.sig.nunique())}
       for s, g in X.groupby('seg4')}
cov_note = '分母＝D=S／D>S 官方訊號日（含 2026-10-12 latest）的 C0 股池股月；「有紀錄」只代表來源有該檔該營收月的列，不代表可判定（Tasi 首次加入檔案、新聞數值不符等在分類時另計）。MOPS 官方頁與 archive.org 封存：連不到（CONNECT 403），覆蓋 0。'

# ---------- 一致性閘門 ----------
def tp(s): return pd.to_datetime(s, utc=True, errors='coerce').dt.tz_convert('Asia/Taipei')
T = FS[FS.source == 'tasi'].copy(); T['Tt'] = tp(T['T']); T['Tp'] = tp(T['T_prev'])
U = FS[FS.source == 'upstream'].copy(); U['Tt'] = tp(U['T']); U['Tp'] = tp(U['T_prev'])
A = AN.copy(); A['a'] = pd.to_datetime(A.ann_date)
def iv_ok(df):
    m = df.merge(A, on=['code', 'ym'])
    m = m[m.Tp.notna()]
    lo = m.Tp.dt.tz_localize(None).dt.normalize(); hi = m.Tt.dt.tz_localize(None).dt.normalize()
    ok = (m.a >= lo) & (m.a <= hi)
    return {'n_with_prev': int(len(m)), 'agree_interval_contains_ann_date': round(float(ok.mean()), 4) if len(m) else None,
            'n_T_before_ann_date': int((hi < m.a).sum())}
gate_tasi_mops = iv_ok(T); gate_up_mops = iv_ok(U)
tu = T.merge(U, on=['code', 'ym'], suffixes=('_t', '_u'))
tu = tu[tu.Tp_t.notna() & tu.Tp_u.notna()]
ovl = (tu.Tp_t <= tu.Tt_u) & (tu.Tp_u <= tu.Tt_t)
gate_tasi_up = {'n_both_with_prev': int(len(tu)), 'interval_overlap': round(float(ovl.mean()), 4) if len(tu) else None}

# 新聞數值閘門：提及值對 tw_revenue（個別）或 rev_5y 營收（千元）在有效位數容差內
tw = pd.read_csv('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv', dtype={'code': str})
tw['ym'] = tw.month.str[:7]; tw = tw.drop_duplicates(['code', 'ym'])
r5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype={'code': str})
TR = dict(zip(zip(tw.code, tw.ym), tw.revenue_k_twd)); TR5 = dict(zip(zip(r5.code, r5.ym), r5.revenue))
def sig_tol(v):
    if not np.isfinite(v) or v <= 0: return np.nan
    s = str(int(round(v))); z = len(s) - len(s.rstrip('0'))
    return 0.5 * 10 ** z + 1.0
def news_match(row, tol_mode):
    v = row.rev_k
    tv = [x for x in (TR.get((row.code, row.ym)), TR5.get((row.code, row.ym))) if x is not None and np.isfinite(x)]
    if not tv or not np.isfinite(v): return 'NO_TRUTH'
    if tol_mode == 'sig':
        tol = sig_tol(v); return 'MATCH' if any(abs(v - t) <= tol for t in tv) else 'MISMATCH'
    return 'MATCH' if any(abs(v / t - 1) <= 0.15 for t in tv if t > 0) else 'MISMATCH'
NW['m_sig'] = [news_match(r, 'sig') for r in NW.itertuples()]
NW['m_15'] = [news_match(r, '15') for r in NW.itertuples()]
NW['tsd'] = pd.to_datetime(NW.ts)
NW['lagm'] = [(pd.Period(t, 'M') - pd.Period(y, 'M')).n for t, y in zip(NW.tsd, NW.ym)]
g = NW[NW.lagm == 1]
gate_news = {'n_mentions': int(len(NW)), 'n_published_next_month': int(len(g)),
             'by_src': {s: {'n': int(len(x)), 'match_sig': round(float((x.m_sig == 'MATCH').mean()), 4), 'match_15pct': round(float((x.m_15 == 'MATCH').mean()), 4),
                            'no_truth': round(float((x.m_sig == 'NO_TRUTH').mean()), 4), 'has_time': round(float(x.has_time.astype(str).eq('True').mean()), 4)}
                        for s, x in g.groupby('src')},
             'note': '只看「次月發布」的提及（營收月＋1 個月內）；MATCH＝提及值與 tw_revenue 個別營收或 rev_5y 營收在有效位數容差（末位 ±0.5）內。鉅亨多報合併營收，個別營收不符時不採用。'}

# ---------- 探索：D<S 日的 Tasi 首見時點（機制可見性；不涉 D=S 判定） ----------
Y = PL[(PL.case == 'D<S') & (PL.seg4 == 'S2406')].copy()
Tt = T.set_index(['code', 'ym'])
def first_rel(code, ym, sig, D):
    try: r = Tt.loc[(code, ym)]
    except KeyError: return None
    t = r['Tt']
    if pd.isna(t): return None
    return (t.tz_localize(None).normalize() - pd.Timestamp(D)).days
DDi = DD.set_index('sig')
Y['rel_to_D'] = [first_rel(c, y, s, DDi.at[s, 'D_engine']) for c, y, s in zip(Y.code, Y.rev_month, Y.sig)]
yy = Y[Y.rel_to_D.notna()]
early = yy[yy.rel_to_D < 0]; late = yy[yy.rel_to_D >= 0]
expl_mech = {'scope': '2024-06 起 D<S 訊號日的 C0 股池（Tasi 首見日相對期限 D 的日數；首見＝18:20 快照）',
             'n': int(len(yy)), 'share_first_seen_before_D': round(float((yy.rel_to_D < 0).mean()), 4),
             'pr_rev_mean_seen_before_D': round(float(early.pr_rev.mean()), 4), 'pr_rev_mean_seen_on_or_after_D': round(float(late.pr_rev.mean()), 4),
             'diff': round(float(early.pr_rev.mean() - late.pr_rev.mean()), 4),
             'note': '只描述「好消息早公告」在 C0 股池是否看得到（o11 P2 已知 +0.078）；不涉及任何 D=S 槽位判定'}

out = {'utc': pd.Timestamp.utcnow().isoformat(), 'no_returns': '只看覆蓋與時間戳；未看名次變化、未算報酬、未算任何判定比例',
       'disk': DISK, 'connectivity': (RAW / 'step0_connectivity.txt').read_text(encoding='utf-8'),
       'proxy_failures': (RAW / 'step0_proxy_failures.txt').read_text(encoding='utf-8'),
       'blocked_hosts_for_center': ['mops.twse.com.tw', 'mopsov.twse.com.tw', 'web.archive.org', 'www.twse.com.tw', 'www.tpex.org.tw'],
       'reachable': ['github.com（git ls-remote 成功：Tasi HEAD 0e9ed7ad、Intern-in-CTBC 258f42aa）', '本機上游 repo（只有 1 個 commit；完整首見已由 o11 整理）'],
       'coverage_by_seg4': cov, 'coverage_note': cov_note,
       'gate_tasi_vs_mops': gate_tasi_mops, 'gate_upstream_vs_mops': gate_up_mops, 'gate_tasi_vs_upstream': gate_tasi_up, 'gate_news_value': gate_news,
       'explore_mechanism_DltS': expl_mech, 'raw_manifest': man, 'runtime_s': round(time.time() - T0, 1)}
json.dump(out, open(HERE / 'explore.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: out[k] for k in ('disk', 'coverage_by_seg4', 'gate_tasi_vs_mops', 'gate_upstream_vs_mops', 'gate_tasi_vs_upstream', 'gate_news_value', 'explore_mechanism_DltS')}, ensure_ascii=False, indent=1, default=str))
