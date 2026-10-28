#!/usr/bin/env python3
"""o_v6_price_panel_official 共用常數與工具（只讀其他資料夾；不改任何既有檔案）。"""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = HERE / 'raw'
DATA = HERE / 'data'
PANEL4 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
FIELDS = ['adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src']
HSCHI = Path('/home/user/hschi1106/stock_ga/data/stock_dayK')
HAWAN = Path('/home/user/hawan0228/fintech_ai/data/external/raw/prices')
HENRY = RAW / 'henry' / 'stocks'
TASI = RAW / 'tasi' / 'kbar'
HIST = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/history.csv.gz')
XTR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
E50F = Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')
TAIEX = Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')
V4DIR = AG / 'o_v4_revenue_patch_official'
SPADIR = AG / 'o_main_spa_breaks'

# V6 替換窗（事前寫死）：2024-01-02～2025-09-23（src 3 期間＋2024-01 缺口）
W_A, W_B = pd.Timestamp('2024-01-02'), pd.Timestamp('2025-09-23')
GAP_A, GAP_B = pd.Timestamp('2024-01-02'), pd.Timestamp('2024-02-14')
# 新 src 碼
SRC_FINMIND, SRC_TASI, SRC_OTHER, SRC_Y5M = 5, 6, 7, 3

def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def load_panel(d=PANEL4, fields=FIELDS):
    return {k: pd.read_parquet(Path(d) / f'{k}.parquet') for k in fields}

def taiex_cal():
    t = pd.read_csv(TAIEX, parse_dates=['date'])['date']
    return pd.DatetimeIndex(sorted(set(t)))

def e50():
    return pd.read_csv(E50F, parse_dates=['date']).set_index('date')['adj'].sort_index()

def jdump(obj, f):
    Path(f).write_text(json.dumps(obj, ensure_ascii=False, indent=1,
                                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding='utf-8')

def gaps(dates, cal, a=None, b=None, min_len=6):
    """dates 中相對交易日曆 cal 連續缺 ≥ min_len（即 > 5）個交易日的區段。"""
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
