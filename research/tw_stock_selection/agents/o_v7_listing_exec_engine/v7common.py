#!/usr/bin/env python3
"""o_v7_listing_exec_engine 共用常數與工具（只讀其他資料夾；不改任何既有檔案）。
raw/ 下的資料一律當不可信 CSV 讀。"""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = HERE / 'raw'
DATA = HERE / 'data'
V6DIR = AG / 'o_v6_price_panel_official'
V6DATA = V6DIR / 'data'
V4DIR = AG / 'o_v4_revenue_patch_official'
SPADIR = AG / 'o_main_spa_breaks'
PANEL4 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
FIELDS = ['adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src']
E50F = Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')
TAIEX = Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')
XTR = RAW / 'screener' / 'extras'
HIST = RAW / 'screener' / 'history.csv.gz'
SLIST = RAW / 'screener' / 'stock_list.csv'
REV = RAW / 'rev' / 'tw_revenue.csv'
TASI = RAW / 'tasi'
W_A, W_B = pd.Timestamp('2024-01-02'), pd.Timestamp('2025-09-23')
GAP_A, GAP_B = pd.Timestamp('2024-01-02'), pd.Timestamp('2024-02-14')
TYPHOON = [pd.Timestamp(x) for x in ('2024-07-24', '2024-07-25', '2024-10-02', '2024-10-03', '2024-10-31')]
SRC_FINMIND, SRC_TASI, SRC_OTHER, SRC_Y5M = 5, 6, 7, 3

def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def load_panel(d, fields=FIELDS):
    return {k: pd.read_parquet(Path(d) / f'{k}.parquet') for k in fields}

def taiex_cal():
    t = pd.read_csv(TAIEX, parse_dates=['date'])['date']
    return pd.DatetimeIndex(sorted(set(t)))

def e50():
    return pd.read_csv(E50F, parse_dates=['date']).set_index('date')['adj'].sort_index()

def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [conv(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, pd.Timestamp): return str(o.date())
    return o

def jdump(obj, f):
    Path(f).write_text(json.dumps(conv(obj), ensure_ascii=False, indent=1, default=str), encoding='utf-8')

def jload(f):
    f = Path(f)
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}

def gaps(dates, cal, a=None, b=None, min_len=6):
    dates = pd.DatetimeIndex(dates)
    a = dates.min() if a is None else pd.Timestamp(a); b = dates.max() if b is None else pd.Timestamp(b)
    c = cal[(cal >= a) & (cal <= b)]
    miss = ~c.isin(dates)
    out = []; i = 0; n = len(c)
    while i < n:
        if miss[i]:
            j = i
            while j + 1 < n and miss[j + 1]: j += 1
            if j - i + 1 >= min_len:
                out.append({'from': str(c[i].date()), 'to': str(c[j].date()), 'n_trading_days': int(j - i + 1)})
            i = j + 1
        else:
            i += 1
    return out

RES_F = HERE / 'results.json'
def load_res(): return jload(RES_F)

def check_prereg():
    res = load_res()
    assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：先跑 prereg.py'
    assert canon_sha(res['PREREG']) == res['PREREG_sha256'], 'PREREG 被改過'
    for p, h in res['PREREG']['hashes'].items():
        assert sha(p) == h, f'{p} 與 PREREG 登記的 sha256 不符'
    return res

def save_key(key, obj, require_prereg=True):
    res = load_res()
    if require_prereg: assert 'PREREG' in res
    res[key] = conv(obj)
    jdump(res, RES_F)
