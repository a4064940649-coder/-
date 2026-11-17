"""S66 資料載入：官方日 K（raw/jb7k6295r7-bot_tw-stock-data 的 data/universe/daily，git cat-file 唯讀、不連網）裡所有 00 開頭代號。
只讀；輸出到暫存 TMP/etf_daily.parquet。"""
import io, os, subprocess, sys
from pathlib import Path
import pandas as pd
RAW = Path('/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data')
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s66')

def blobs(sub):
    env = dict(os.environ, GIT_NO_LAZY_FETCH='1')
    ls = subprocess.run(['git', '-C', str(RAW), 'ls-tree', 'HEAD', sub + '/'], capture_output=True, text=True, check=True, env=env).stdout
    items = [(l.split()[2], l.split('\t')[1]) for l in ls.strip().split('\n') if l.endswith('.csv')]
    p = subprocess.Popen(['git', '-C', str(RAW), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    for sha, path in items:
        p.stdin.write((sha + '\n').encode()); p.stdin.flush()
        hdr = p.stdout.readline().decode().split()
        if len(hdr) < 3 or hdr[1] != 'blob':
            yield path, None; continue
        n = int(hdr[2]); data = p.stdout.read(n); p.stdout.read(1)
        yield path, io.BytesIO(data)
    p.stdin.close(); p.wait()

def build():
    TMP.mkdir(parents=True, exist_ok=True)
    fr, miss = [], []
    for path, fh in blobs('data/universe/daily'):
        if fh is None:
            miss.append(path); continue
        d = pd.read_csv(fh, dtype={'stock_id': str}, usecols=['date', 'stock_id', 'name', 'market', 'open', 'close', 'volume', 'amount', 'change'])
        fr.append(d[d.stock_id.str.match(r'^00\d{2,4}[A-Z]?$')])
    d = pd.concat(fr, ignore_index=True)
    for c in ['open', 'close', 'volume', 'amount', 'change']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d.to_parquet(TMP / 'etf_daily.parquet')
    (TMP / 'missing_blobs.txt').write_text('\n'.join(miss))
    return d, miss

if __name__ == '__main__':
    d, miss = build()
    print(d.shape, d.date.min(), d.date.max(), d.stock_id.nunique(), 'missing blobs', len(miss))
