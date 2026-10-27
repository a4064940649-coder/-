# -*- coding: utf-8 -*-
"""Second independent source for SEO/CB: news mirror aRthur08701244/Intern-in-CTBC (鉅亨/MoneyDJ/MOPS headlines,
2013-2021, ~100 items per day; a SAMPLE, not complete). Read-only from o_index_deletion_flow/raw (untrusted text).
Extract board-resolution announcements of 現金增資 and 轉換公司債 for TWSE-listed issuers, then measure
(a) share found in events.csv (FSC/TPEx) -> sampled coverage, (b) lag board -> FSC filing / TPEx listing.
No returns are used. Output: news_events.csv, news_xcheck.json
"""
import os, re, glob, json, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
O1 = os.path.join(HERE, '..', 'o_index_deletion_flow')
sys.path.insert(0, O1)
import namemap   # read-only import (name -> code)
sys.path.pop(0)
NM = namemap.build()
NEWS = os.path.join(O1, 'raw', 'aRthur08701244_Intern-in-CTBC', 'AI_Investment', 'OpenIEStanza_EventEmbedding', 'input', 'EB_EventExtraction2')
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
C = pd.read_parquet(os.path.join(P, 'close_raw.parquet'))
COLS = set(C.columns)

EXCL = ('初次上市', '初次上櫃', '上市前', '上櫃前', '子公司', '孫公司', '私募', '繳款', '除權', '發放', '更正', '補充', '代收', '存儲',
        '洽特定人', '放棄認購', '不執行', '取消', '喊卡', '停止', '庫藏股', '減資', '承銷價', '撤回', '競拍', '公開申購', '掛牌', '轉換價格', '贖回', '收回', '到期', '變更', '基準日', '停止過戶')
rows = []
for f in sorted(glob.glob(os.path.join(NEWS, '*.csv'))):
    try:
        x = pd.read_csv(f, usecols=['News', 'All'])
    except Exception:
        continue
    for t, a in zip(x['News'].astype(str), x['All'].astype(str)):
        head = a[:110]
        is_seo = ('現金增資' in head or '現增' in head) and ('董事會' in head or '決議' in head or '擬辦理' in head or '擬現增' in head)
        is_cb = ('轉換公司債' in head or '可轉債' in head or '可轉換' in head or re.search(r'(?<![A-Za-z])E?CB(?![A-Za-z])', head) is not None) and ('董事會' in head or '決議' in head or '擬發行' in head or '擬發' in head or '籌資' in head)
        if not (is_seo or is_cb):
            continue
        if any(k in head for k in EXCL):
            continue
        d = pd.Timestamp(t[:8])
        m = re.search(r'董事會決議日期[:：]\s*(\d{2,3})/(\d{1,2})/(\d{1,2})', a[:400])
        if m:
            try:
                d = pd.Timestamp(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3)))
            except Exception:
                pass
        code = None
        m = re.match(r'^\*?([^：:\s]{2,8})[：:]', head)
        if m:
            code = NM.get(namemap.norm(m.group(1)))
        if code is None:
            m = re.search(r'[（(](\d{4})[)）]', a[:200])
            if m:
                code = m.group(1)
        if code is None:
            for k in sorted(NM, key=len, reverse=True):
                if len(k) >= 2 and head.startswith(k):
                    code = NM[k]; break
        if code is None or code not in COLS:
            continue
        s = C[code]
        w = s[(s.index >= d - pd.Timedelta(days=7)) & (s.index <= d + pd.Timedelta(days=7))]
        if not w.notna().any():
            continue   # not TWSE-listed at the time (panel = TWSE)
        rows.append(dict(date=d, code=code, kind='SEO' if is_seo and not is_cb else ('CB' if is_cb and not is_seo else 'BOTH'), head=head[:80].replace('\n', ' ')))
N = pd.DataFrame(rows).drop_duplicates(['code', 'kind', 'date'])
# collapse same code/kind within 30 days (follow-up articles)
N = N.sort_values(['code', 'kind', 'date'])
keep = []; last = {}
for i, r in N.iterrows():
    k = (r['code'], r['kind'])
    if k in last and (r['date'] - last[k]).days <= 30:
        continue
    last[k] = r['date']; keep.append(i)
N = N.loc[keep].reset_index(drop=True)
N.to_csv(os.path.join(HERE, 'news_events.csv'), index=False)

E = pd.read_csv(os.path.join(HERE, 'events.csv'), dtype={'code': str}, parse_dates=['recv', 'eff', 'list_date', 'anchor'])
summ = json.load(open(os.path.join(HERE, 'events_build_summary.json')))
cov = {int(k): (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in summ['fsc_year_coverage'].items()}

def covered(d):
    y = d.year
    return y in cov and cov[y][0] - pd.Timedelta(days=5) <= d <= cov[y][1] - pd.Timedelta(days=60)

FA = pd.read_csv(os.path.join(HERE, 'fsc_cases_all.csv'), dtype={'code': str}, parse_dates=['recv', 'eff'])
out = {}
for kind in ('SEO', 'CB'):
    sub = N[N['kind'].isin([kind, 'BOTH'])].copy()
    res = []
    for _, r in sub.iterrows():
        if kind == 'SEO':
            c = FA[(FA['code'] == r['code']) & FA['item'].astype(str).str.startswith('現金') & (FA['recv'] >= r['date'] - pd.Timedelta(days=150)) & (FA['recv'] <= r['date'] + pd.Timedelta(days=200))]
            lag = (c['recv'].min() - r['date']).days if len(c) else np.nan
            cv = covered(r['date'])
        else:
            c = E[(E['code'] == r['code']) & (E['etype'] == 'CB') & (E['list_date'] >= r['date'] - pd.Timedelta(days=120)) & (E['list_date'] <= r['date'] + pd.Timedelta(days=240))]
            lag = (c['list_date'].min() - r['date']).days if len(c) else np.nan
            cv = r['date'] <= pd.Timestamp('2026-02-01')
        res.append(dict(date=r['date'], code=r['code'], found=len(c) > 0, lag=lag, covered=cv, head=r['head'], fsc_any=len(FA[(FA['code'] == r['code']) & (FA['recv'] >= r['date'] - pd.Timedelta(days=150)) & (FA['recv'] <= r['date'] + pd.Timedelta(days=200))]) > 0))
    R = pd.DataFrame(res)
    Rc = R[R['covered']]
    out[kind] = dict(n_news=int(len(R)), n_in_covered_window=int(len(Rc)), coverage=float(Rc['found'].mean()) if len(Rc) else None,
                     lag_days_p10_p50_p90=[float(x) for x in np.nanpercentile(Rc['lag'].dropna(), [10, 50, 90])] if Rc['lag'].notna().any() else None,
                     not_found_examples=Rc[~Rc['found']][['date', 'code', 'head']].astype(str).head(25).values.tolist(),
                     by_year=Rc.groupby(Rc['date'].dt.year)['found'].agg(['size', 'mean']).round(3).to_dict(orient='index'))
    R.to_csv(os.path.join(HERE, f'news_xcheck_{kind}.csv'), index=False)
json.dump(out, open(os.path.join(HERE, 'news_xcheck.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])
