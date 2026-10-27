#!/usr/bin/env python3
"""診斷 (d) 的價格修補（不看報酬）：panel 2019-05-17～2023 只有存活股；最後交易日停在 2019-05-16 的股票，
用 taiwanan tse_trading_raw_data（2015-01～2021-04 證交所個股日成交，含之後下市者）補 2019-05-17 之後的價量。
日報酬＝收盤 ÷ 參考價 − 1，參考價＝收盤 ± |漲跌價差| 中最接近前收者（來源部分期間掉負號）（漲跌價差相對當日參考價，除權息日也成立，近似含息報酬）。
交叉核對：2015-01～2019-05-16 與 panel 重疊日逐筆比對收盤（相等）與日報酬（差 ≤ 0.1pp）。
輸出 price_patch.parquet（長表 date,code,open,high,close,value,ret）＋ price_patch_meta.json（sha256）。
"""
import subprocess, csv, re, json, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE / 'raw' / 'taiwanan_TW_Stock_Exchange_Crawler'
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
C = pd.read_parquet(P / 'close_raw.parquet'); RET = pd.read_parquet(P / 'ret.parquet')
lt = C.apply(lambda s: s.last_valid_index())
CUT = pd.Timestamp('2019-05-16')
targets = sorted(lt.index[lt == CUT])

def git(*a): return subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, check=True).stdout
def num(x):
    x = re.sub(r'[^0-9.\-]', '', x.replace(',', ''))
    try: return float(x)
    except ValueError: return np.nan
rows = []; got = {}
for c in targets:
    path = f'tse_trading_raw_data/{c}.csv'
    # 最後存在的版本
    h = git('log', '-1', '--format=%H', '--', path).decode().strip()
    if not h: continue
    try: raw = git('show', f'{h}:{path}')
    except subprocess.CalledProcessError: raw = git('show', f'{h}^:{path}')
    got[c] = hashlib.sha256(raw).hexdigest()
    for r in csv.reader(raw.decode('utf-8-sig', errors='replace').splitlines()[2:]):
        if len(r) < 9 or not re.fullmatch(r'\d{2,3}/\d{2}/\d{2}', r[0].strip()): continue
        y, m, dd = r[0].strip().split('/')
        dt = pd.Timestamp(int(y) + 1911, int(m), int(dd))
        o, hi, cl, chg, val = num(r[2]), num(r[3]), num(r[5]), num(r[6]), num(r[8])
        rows.append((dt, c, o, hi, cl, val, chg))
D = pd.DataFrame(rows, columns=['date', 'code', 'open', 'high', 'close', 'value', 'chg']).drop_duplicates(['date', 'code'], keep='last').sort_values(['code', 'date'])
# 這個爬蟲在部分期間把負號弄丟（漲跌價差只剩絕對值），所以參考價取「收盤 ± |漲跌價差|」中最接近前一筆收盤者
pc = D.groupby('code')['close'].shift(1)
a = D['chg'].abs()
c1, c2 = D['close'] - a, D['close'] + a
ref = np.where(np.abs(c1 - pc) <= np.abs(c2 - pc), c1, c2)
ref = np.where(pc.isna(), c1, ref)
D['ret'] = np.where((ref > 0) & np.isfinite(D['close']), D['close'] / ref - 1, np.nan)
# 交叉核對
ov = D[D['date'] <= CUT].copy()
ov['c_panel'] = [C.at[d, c] if d in C.index else np.nan for d, c in zip(ov['date'], ov['code'])]
ov['r_panel'] = [RET.at[d, c] if d in RET.index else np.nan for d, c in zip(ov['date'], ov['code'])]
m = ov['c_panel'].notna() & ov['close'].notna()
close_agree = float((np.abs(ov.loc[m, 'close'] - ov.loc[m, 'c_panel']) < 1e-6).mean())
m2 = m & ov['r_panel'].notna() & ov['ret'].notna()
ret_agree = float((np.abs(ov.loc[m2, 'ret'] - ov.loc[m2, 'r_panel']) <= 0.001).mean())
patch = D[D['date'] > CUT][['date', 'code', 'open', 'high', 'close', 'value', 'ret']]
patch = patch[np.isfinite(patch['close']) & (patch['close'] > 0)]
f = HERE / 'price_patch.parquet'; patch.to_parquet(f)
end = patch.groupby('code')['date'].max().astype(str).to_dict()
meta = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'targets_lastday_2019_05_16': targets,
        'n_targets': len(targets), 'n_found': len(got), 'not_found': [c for c in targets if c not in got],
        'overlap_close_agree': close_agree, 'overlap_ret_agree_0.1pp': ret_agree, 'n_overlap_days': int(m.sum()), 'n_overlap_ret': int(m2.sum()),
        'patch_rows': int(len(patch)), 'patch_end_by_code': end, 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'src_sha256': got}
(HERE / 'price_patch_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in meta.items() if k != 'src_sha256'}, ensure_ascii=False, indent=1))
