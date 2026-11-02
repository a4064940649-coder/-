"""獨立複製 o10 第 0 步：從 raw git 封存讀出日 K、本益比（上市/上櫃）、季報、月營收 → 本資料夾 data/*.parquet"""
import subprocess, io, glob
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'data'
RAW = Path('/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data')


def git_files(sub):
    ls = subprocess.run(['git', '-C', str(RAW), 'ls-tree', 'HEAD', sub + '/'], capture_output=True, text=True, check=True).stdout
    rows = []
    for ln in ls.splitlines():
        meta, path = ln.split('\t')
        _, typ, sha = meta.split()
        if typ == 'blob' and path.endswith('.csv') and Path(path).name[:4].isdigit():
            rows.append((path, sha))
    return rows


def cat_all(rows, usecols, dtype):
    p = subprocess.Popen(['git', '-C', str(RAW), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = []
    for path, sha in rows:
        p.stdin.write((sha + '\n').encode()); p.stdin.flush()
        hdr = p.stdout.readline().split()
        n = int(hdr[2])
        b = p.stdout.read(n); p.stdout.read(1)
        if n < 20:
            continue
        try:
            d = pd.read_csv(io.BytesIO(b), dtype=dtype, usecols=lambda c: c in usecols)
        except Exception as e:
            print('ERR', path, e); continue
        out.append(d)
    p.stdin.close(); p.wait()
    return pd.concat(out, ignore_index=True)


def main():
    # 日 K
    rows = git_files('data/universe/daily')
    d = cat_all(rows, {'date', 'stock_id', 'market', 'open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'limit'},
                {'stock_id': str, 'date': str, 'market': str, 'limit': str})
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'change']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[d.stock_id.str.fullmatch(r'\d{4,6}[A-Z]?') | d.stock_id.isin(['0050'])]
    d.to_parquet(OUT / 'daily.parquet', index=False)
    print('daily', d.shape, d.date.min(), d.date.max())
    # 本益比
    allp = []
    for sub, mk in [('data/early/per', 'twse'), ('data/universe/per', 'twse'), ('data/early/otcper', 'tpex'), ('data/universe/otcper', 'tpex')]:
        rows = git_files(sub)
        p = cat_all(rows, {'date', 'stock_id', 'close', 'per', 'pbr', 'fs_quarter'},
                    {'stock_id': str, 'date': str, 'fs_quarter': str})
        p['mkt'] = mk
        allp.append(p)
        print(sub, p.shape)
    p = pd.concat(allp, ignore_index=True)
    for c in ['close', 'per', 'pbr']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    p = p[p.stock_id.str.fullmatch(r'\d{4}')]
    p.to_parquet(OUT / 'per.parquet', index=False)
    # 季報
    fs = sorted(glob.glob(str(RAW / 'data/mops/fin_hist/*.csv')))
    f = pd.concat([pd.read_csv(x, dtype={'stock_id': str}) for x in fs], ignore_index=True)
    f.to_parquet(OUT / 'fin_raw.parquet', index=False)
    print('fin', f.shape)
    fs = sorted(glob.glob(str(RAW / 'data/mops/revenue_hist/*.csv')))
    r = pd.concat([pd.read_csv(x, dtype={'stock_id': str}) for x in fs], ignore_index=True)
    r = r.rename(columns={'當月營收': 'rev'})[['stock_id', 'period', 'market', 'rev']]
    r['rev'] = pd.to_numeric(r.rev, errors='coerce')
    r.to_parquet(OUT / 'rev_m.parquet', index=False)
    print('rev', r.shape)


if __name__ == '__main__':
    main()
