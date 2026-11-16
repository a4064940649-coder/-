#!/usr/bin/env python3
"""S39 第 1 步：從公開 repo shohokuno10/Tasi（MOPS 上市營收彙總表每日快照）抽每檔每月的首次值 FP、窗末值 TL、
首次快照中的上月營收 V1／去年當月營收 V12；並和 o_rev_event_clock_ts/data/firstseen_sources.csv.gz 的首次出現時間核對。
clone 當不可信資料：只用 git log／git show 讀，不執行其中任何程式。輸出 data/first_print.csv.gz、extract_check.json。
用法：python3 -I -B extract.py（先 git clone --filter=blob:none --no-checkout https://github.com/shohokuno10/Tasi 到 scratchpad/s39/tasi）"""
import io, json, subprocess, datetime
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
AG = H.parent
TMP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s39')
REPO = TMP / 'tasi'
MONTHS = [str(p) for p in pd.period_range('2024-06', '2026-09', freq='M')]


def git(*a, binary=False):
    r = subprocess.run(['git', '-c', 'core.quotepath=off', '-C', str(REPO), *a], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode('utf-8', 'replace')


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '').str.strip(), errors='coerce')


def parse(b):
    df = pd.read_csv(io.BytesIO(b), dtype=str)
    c0 = [c for c in df.columns if '代號' in c][0]
    g = lambda k: [c for c in df.columns if c.strip() == k][0]
    out = pd.DataFrame(dict(code=df[c0].astype(str).str.strip().str.strip('"').str.strip(),
                            cur=num(df[g('當月營收')]), prev=num(df[g('上月營收')]), ly=num(df[g('去年當月營收')])))
    out = out[out['cur'].notna() & (out['code'] != '')]
    return out.drop_duplicates('code', keep='first').set_index('code')


rows, chk = [], dict(repo='https://github.com/shohokuno10/Tasi', head=git('rev-parse', 'HEAD').strip(), per_month={})
for ym in MONTHS:
    f = f'營收/{ym[:4]}{ym[5:]}.csv'
    cm = [l.split() for l in git('log', '--format=%H %cI', '--', f).splitlines() if l.strip()][::-1]
    first, last, nchg, prevv = {}, None, {}, {}
    for h, t in cm:
        try:
            d = parse(git('show', f'{h}:{f}', binary=True))
        except Exception:
            continue
        for c, r in d.iterrows():
            if c not in first:
                first[c] = (r['cur'], r['prev'], r['ly'], t, h)
            elif prevv.get(c) is not None and r['cur'] != prevv[c]:
                nchg[c] = nchg.get(c, 0) + 1
            prevv[c] = r['cur']
        last = d
    for c, (v, p, ly, t, h) in first.items():
        rows.append((c, ym, v, p, ly, t, h, last['cur'].get(c, np.nan) if last is not None else np.nan, nchg.get(c, 0)))
    chk['per_month'][ym] = dict(n_commits=len(cm), first_commit=cm[0][1] if cm else None, last_commit=cm[-1][1] if cm else None, n_codes=len(first),
                                n_changed_in_window=int(sum(1 for c in nchg)))
    print(ym, len(cm), len(first), flush=True)

FP = pd.DataFrame(rows, columns=['code', 'ym', 'FP', 'V1', 'V12', 'T_fp', 'commit_fp', 'TL', 'n_chg_window'])
# 核對 firstseen_sources 的首次出現 commit／時間
fs = pd.read_csv(AG / 'o_rev_event_clock_ts' / 'data' / 'firstseen_sources.csv.gz', dtype={'code': str})
fs = fs[(fs['source'] == 'tasi') & fs['ym'].isin(MONTHS)]
mg = FP.merge(fs[['code', 'ym', 'T', 'commit']], on=['code', 'ym'], how='outer', indicator=True)
both = mg[mg['_merge'] == 'both']
chk['firstseen_sources_columns'] = list(pd.read_csv(AG / 'o_rev_event_clock_ts' / 'data' / 'firstseen_sources.csv.gz', nrows=1).columns)
chk['firstseen_sources_has_values'] = False
chk['crosscheck_firstseen'] = dict(n_fp=int(len(FP)), n_fs=int(len(fs)), n_both=int(len(both)), only_fp=int((mg['_merge'] == 'left_only').sum()),
                                   only_fs=int((mg['_merge'] == 'right_only').sum()), commit_agree=float((both['commit'] == both['commit_fp']).mean()),
                                   T_agree=float((both['T'] == both['T_fp']).mean()))
FP.to_csv(H / 'data' / 'first_print.csv.gz', index=False)
(H / 'extract_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
print(json.dumps(chk['crosscheck_firstseen'], ensure_ascii=False))
