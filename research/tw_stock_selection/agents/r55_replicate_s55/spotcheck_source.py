#!/usr/bin/env python3
"""R55 抽查：從 asoul_source_manifest.json 指向的原始來源檔（Asoul/tsec 本機副本）隨機抽 20 檔 × 5 天，核對長表的收、開、成交金額；
並核對這 20 個來源檔的 sha256 與 manifest 一致、長表 sha256 與 SHA256SUMS.json 一致。另做整檔核對（這 20 檔在 2007-01-02～2015-06-30 的全部列）。
python3 -I -B spotcheck_source.py → scratchpad/r55/spotcheck.json、spotcheck_rows.csv"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

RS = Path('/home/user/-/research/tw_stock_selection')
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r55')
man = json.loads((RS / 'raw/pre2015/asoul_source_manifest.json').read_text())
sums = json.loads((RS / 'raw/pre2015/SHA256SUMS.json').read_text())
src = Path(man['path'])
LONG = RS / 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = dict(long_sha_ok=sha(LONG) == sums['asoul_twse_2007_2015H1.csv.gz'] == man['derived_sha256'], n_manifest_files=len(man['files']))
d = pd.read_csv(LONG, dtype={'code': str})
rng = np.random.default_rng(20551117)
vc = d['code'].value_counts()
codes = sorted(set(vc[vc >= 5].index) & set(man['files']))
pick = sorted(rng.choice(codes, 20, replace=False).tolist())
num = lambda s: pd.to_numeric(s.astype(str).str.replace(',', '').str.strip(), errors='coerce')
rows = []; full = {}
for c in pick:
    f = src / f'{c}.csv'
    ok_sha = sha(f) == man['files'][c]
    s = pd.read_csv(f, header=None, dtype=str, names=['d', 'vol', 'amt', 'o', 'h', 'l', 'c', 'chg', 'n'])
    p = s['d'].str.strip().str.split('/', expand=True)
    s['date'] = pd.to_datetime(dict(year=p[0].astype(int) + 1911, month=p[1].astype(int), day=p[2].astype(int)), errors='coerce').dt.strftime('%Y-%m-%d')
    for k in ('amt', 'o', 'c'):
        s[k] = num(s[k])
    s = s.drop_duplicates('date', keep='last').set_index('date')
    g = d[d['code'] == c].set_index('date')
    days = sorted(rng.choice(g.index.to_numpy(), 5, replace=False).tolist())
    for x in days:
        a = g.loc[x]; b = s.loc[x] if x in s.index else None
        eq = lambda u, v: (pd.isna(u) and pd.isna(v)) or (pd.notna(u) and pd.notna(v) and abs(float(u) - float(v)) < 1e-6)
        rows.append(dict(code=c, date=x, src_sha_ok=ok_sha, long_close=a['close'], src_close=None if b is None else b['c'], long_open=a['open'],
                         src_open=None if b is None else b['o'], long_amount=a['amount'], src_amount=None if b is None else b['amt'],
                         match=bool(b is not None and eq(a['close'], b['c']) and eq(a['open'], b['o']) and eq(a['amount'], b['amt']))))
    # 整檔：長表這檔的所有列
    j = g.join(s[['c', 'o', 'amt']], how='left')
    m = ((j['close'] - j['c']).abs() < 1e-6) | (j['close'].isna() & j['c'].isna())
    m &= ((j['open'] - j['o']).abs() < 1e-6) | (j['open'].isna() & j['o'].isna())
    m &= (j['amount'] - j['amt']).abs() < 1e-6
    inrange = s[(s.index >= '2007-01-02') & (s.index <= '2015-06-30')]
    full[c] = dict(src_sha_ok=ok_sha, long_rows=int(len(g)), all_match=bool(m.all()), n_mismatch=int((~m).sum()), src_rows_in_range=int(len(inrange)),
                   src_rows_in_range_not_in_long=int(len(inrange.index.difference(g.index))))
R = pd.DataFrame(rows)
R.to_csv(TMP / 'spotcheck_rows.csv', index=False)
out.update(codes=pick, n_cells=len(R), n_match=int(R['match'].sum()), all_src_sha_ok=bool(R['src_sha_ok'].all()), full_file=full)
(TMP / 'spotcheck.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))
