# 閘門：自行從 git 歷史算 (code, ym) 首見時間戳與合格判定
import subprocess, json, os, io, gzip, sys
import pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, 'raw')
man = json.load(open(os.path.join(HERE, 'manifest.json')))
def blob(repo, b):
    return subprocess.run(['git', '-C', os.path.join(RAW, repo), 'cat-file', 'blob', b], capture_output=True, check=True).stdout
def keys_tasi(path, data):
    if not data.strip(): return set()
    ym = path.split('/')[-1][:4] + '-' + path.split('/')[-1][4:6]
    df = pd.read_csv(io.BytesIO(data), dtype=str)
    c0, cv = df.columns[0], '當月營收'
    df = df[df[cv].notna() & (df[cv].str.strip() != '')]
    return {(c.strip(), ym) for c in df[c0]}
def keys_up(path, data):
    if path.endswith('.gz'): data = gzip.decompress(data)
    if not data.strip(): return set()
    df = pd.read_csv(io.BytesIO(data), dtype=str)
    if 'rev_month' in df.columns:  # revenue.csv：民國年月
        v = df['rev_yoy']; rm = df['rev_month'].str.strip()
        ym = rm.map(lambda s: f'{int(s[:-2]) + 1911}-{s[-2:]}' if isinstance(s, str) and s.isdigit() else None)
    else:
        v = df['revenue']; ym = df['ym']
    ok = v.notna() & (v.astype(str).str.strip() != '') & ym.notna()
    return set(zip(df.loc[ok, 'code'].str.strip(), ym[ok]))
def build(repo, files, parser):
    # 依 commit 時間串起所有檔案版本
    ev = []
    for f, vers in files.items():
        prevT = None
        for v in vers:
            ev.append(dict(file=f, commit=v['commit'], T=pd.Timestamp(v['T']).tz_convert('Asia/Taipei'),
                           Tprev=prevT, blob=v['blob']))
            prevT = pd.Timestamp(v['T']).tz_convert('Asia/Taipei')
    ev.sort(key=lambda e: e['T'])
    first = {}  # key -> dict
    cur = {}
    for e in ev:
        ks = parser(e['file'], blob(repo, e['blob'])) if e['blob'] else set()
        cur[e['file']] = ks
        new = [k for k in ks if k not in first]
        for k in new:
            first[k] = dict(T=e['T'], Tprev=e['Tprev'], file=e['file'], commit=e['commit'])
        # 同一 commit 跨檔：同一 key 若在同一 commit 由多檔新增，取能合格的那一個（Tprev 最近者）
    # 同 commit 其他檔的 Tprev
    bycommit = {}
    for e in ev: bycommit.setdefault(e['commit'], []).append(e)
    for k, r in first.items():
        for e in bycommit[r['commit']]:
            if e['file'] != r['file'] and e['Tprev'] is not None and (r['Tprev'] is None or e['Tprev'] > r['Tprev']):
                if e['blob'] and k in cur_keys_cache(repo, e, parser): r['Tprev'] = e['Tprev']; r['file'] = e['file']
    df = pd.DataFrame([dict(code=k[0], ym=k[1], **v) for k, v in first.items()])
    nper = df.groupby(['commit', 'code']).ym.transform('nunique')
    gap = (df['T'] - df['Tprev']).dt.total_seconds() / 3600
    df['gap_h'] = gap
    df['reason'] = np.where(df.Tprev.isna(), 'file_add', np.where(gap > 36, 'gap36', np.where(nper >= 2, 'batch', 'ok')))
    df['valid'] = df.reason == 'ok'
    return df
_cc = {}
def cur_keys_cache(repo, e, parser):
    k = (repo, e['blob'])
    if k not in _cc: _cc[k] = parser(e['file'], blob(repo, e['blob']))
    return _cc[k]
tasi = build('tasi', {f: v for f, v in man['tasi'].items() if f.endswith('.csv')}, keys_tasi)
up = build('screener', man['screener'], keys_up)
tasi.to_csv(os.path.join(HERE, 'fs_tasi.csv.gz'), index=False); up.to_csv(os.path.join(HERE, 'fs_up.csv.gz'), index=False)
print('tasi keys', len(tasi), tasi.reason.value_counts().to_dict())
print('up keys', len(up), up.reason.value_counts().to_dict(), up.groupby(['file', 'reason']).size().to_dict())
