# -*- coding: utf-8 -*-
"""R0 清潔室：資料層。只讀原始資料（官方日 K git blob、月營收 csv、TWT49U、期限例外表、0050 含息），
不 import 任何既有研究引擎。輸出暫存 npz。用法：python3 -I -B build_data.py <暫存目錄>"""
import sys, os, io, re, gzip, glob, subprocess, csv
import numpy as np, pandas as pd

ROOT = '/home/user/-/research/tw_stock_selection'
REPO = ROOT + '/raw/jb7k6295r7-bot_tw-stock-data'
REV = '/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'
EXR = ROOT + '/raw/twt49u/twse-exright'
ETF = ROOT + '/agents/integrate_o6/data/etf_0050.csv'   # finlab 0050 含息（到 2026-10-22 那一版）
CODE_RE = re.compile(r'^[1-9]\d{3}$')

def main(out):
    env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
    ls = subprocess.run(['git', '-C', REPO, 'ls-tree', 'HEAD', 'data/universe/daily/'], env=env,
                        capture_output=True, text=True, check=True).stdout.splitlines()
    files = []
    for l in ls:
        meta, path = l.split('\t'); sha = meta.split()[2]
        m = re.search(r'(\d{4}-\d{2}-\d{2})\.csv$', path)
        if m: files.append((m.group(1), sha))
    files.sort()
    p = subprocess.Popen(['git', '-C', REPO, 'cat-file', '--batch'], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    rows = []; etf_rows = []
    for d, sha in files:
        p.stdin.write((sha + '\n').encode()); p.stdin.flush()
        hdr = p.stdout.readline().split(); n = int(hdr[2])
        buf = p.stdout.read(n); p.stdout.read(1)
        for r in csv.DictReader(io.StringIO(buf.decode('utf-8'))):
            c = r['stock_id']
            if c == '0050':
                etf_rows.append((d, r['open'], r['close']))
            if r['market'] not in ('twse', 'tpex') or not CODE_RE.match(c): continue
            rows.append((d, c, r['name'], r['market'], r['open'], r['close'], r['amount'], r['change'], r['limit']))
    p.stdin.close(); p.wait()
    df = pd.DataFrame(rows, columns=['date', 'code', 'name', 'market', 'open', 'close', 'amount', 'change', 'limit'])
    for c in ('open', 'close', 'amount', 'change'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    dates = sorted(df['date'].unique()); codes = sorted(df['code'].unique())
    di = {d: i for i, d in enumerate(dates)}; ci = {c: i for i, c in enumerate(codes)}
    T, N = len(dates), len(codes)
    ii = df['date'].map(di).values; jj = df['code'].map(ci).values
    def panel(col, fill=np.nan):
        a = np.full((T, N), fill); a[ii, jj] = df[col].values; return a
    O = panel('open'); C = panel('close'); A = panel('amount'); CH = panel('change')
    present = np.zeros((T, N), bool); present[ii, jj] = True
    twse = np.zeros((T, N), bool); twse[ii, jj] = (df['market'] == 'twse').values
    innov = np.zeros((T, N), bool); innov[ii, jj] = df['name'].str.endswith('創').values
    dr = np.zeros((T, N), bool); dr[ii, jj] = (df['name'].str.contains('DR') | df['code'].str.startswith('91')).values
    name_last = df.groupby('code')['name'].last().reindex(codes).values.astype(str)
    # TWT49U 除權息參考價
    ex_prev = np.full((T, N), np.nan); ex_ref = np.full((T, N), np.nan)
    for f in sorted(glob.glob(EXR + '/*/*.csv.gz')):
        e = pd.read_csv(f, dtype={'code': str})
        for d, c, pc, rp in zip(e['date'], e['code'], e['prev_close'], e['ref_price']):
            if d in di and c in ci and rp > 0:
                ex_prev[di[d], ci[c]] = pc; ex_ref[di[d], ci[c]] = rp
    # 0050
    e0 = pd.DataFrame(etf_rows, columns=['date', 'open', 'close']).set_index('date').astype(float).reindex(dates)
    adj = pd.read_csv(ETF).set_index('date')['adj'].reindex(dates)
    # 月營收
    rv = pd.read_csv(REV, dtype={'code': str})
    rv['month'] = rv['month'].str[:7]
    dup = int(rv.duplicated(['month', 'code']).sum())
    R = rv.drop_duplicates(['month', 'code'], keep='last').pivot(index='month', columns='code', values='revenue_k_twd')
    R = R.reindex(pd.period_range(R.index.min(), R.index.max(), freq='M').astype(str))
    np.savez_compressed(out + '/data.npz', dates=np.array(dates), codes=np.array(codes), O=O, C=C, A=A, CH=CH,
                        present=present, twse=twse, innov=innov, dr=dr, name=name_last, ex_prev=ex_prev, ex_ref=ex_ref,
                        e_open=e0['open'].values, e_close=e0['close'].values, e_adj=adj.values,
                        rev=R.values, rev_months=np.array(R.index), rev_codes=np.array(R.columns), rev_dup=dup)
    print('days', T, dates[0], dates[-1], 'codes', N, 'rev', R.shape, 'dup', dup, 'exright cells', int(np.isfinite(ex_ref).sum()))

if __name__ == '__main__':
    main(sys.argv[1])
