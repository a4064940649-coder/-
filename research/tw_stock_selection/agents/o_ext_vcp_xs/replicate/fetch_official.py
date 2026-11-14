# 獨立複製員：抓 jb7k6295r7-bot/tw-stock-data 的官方日 K（2015 起）做品質抽查與缺口補值敏感度。
# 只輸出到 scratchpad/rep_vcp/，不落地原始檔。用法：python3 -I -B fetch_official.py
import subprocess, io, sys, os, json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, numpy as np
S = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
OUT = S + 'rep_vcp/'
os.makedirs(OUT, exist_ok=True)
codes = pd.read_parquet(S + 'panel/src.parquet').columns.tolist()
URL = 'https://raw.githubusercontent.com/jb7k6295r7-bot/tw-stock-data/main/data/stocks/%s.csv'
def get(c):
    for k in range(3):
        r = subprocess.run(['curl', '-sS', '-m', '60', '-f', URL % c], capture_output=True)
        if r.returncode == 0 and len(r.stdout) > 50:
            try:
                d = pd.read_csv(io.BytesIO(r.stdout), usecols=['date', 'market', 'open', 'high', 'low', 'close', 'volume', 'amount', 'valid_bar'], dtype={'market': str})
                d['code'] = c
                return d
            except Exception as e:
                err = repr(e)
        else:
            err = 'rc=%d %s' % (r.returncode, r.stderr.decode()[:80])
            if 'error: 404' in err: break
    return (c, err)
frames, miss = [], []
with ThreadPoolExecutor(8) as ex:
    for i, r in enumerate(ex.map(get, codes)):
        if isinstance(r, tuple): miss.append(r)
        else: frames.append(r)
        if i % 100 == 0: print(i, len(frames), len(miss), flush=True)
df = pd.concat(frames, ignore_index=True)
df = df[df['date'] <= '2026-10-22']
df.to_parquet(OUT + 'official_long.parquet')
json.dump({'n_codes_ok': len(frames), 'missing': miss, 'rows': int(len(df))}, open(OUT + 'official_fetch.json', 'w'), ensure_ascii=False, indent=1)
print('done', len(frames), len(miss), len(df))
