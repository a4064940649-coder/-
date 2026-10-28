#!/usr/bin/env python3
"""o_main_spa_breaks 共用：換股窗格點、0050、族群規則常數、雜湊工具。
格點＝主策略 AB_V4 的訊號日（o_v4_revenue_patch_official/plan_V4.csv 的 signal 欄）＋評價終點 2026-10-12。
窗報酬＝NAV(下一個訊號日收盤)/NAV(本訊號日收盤)−1（與 v4lib.win_ret 相同定義；簡單報酬，不取對數）。
"""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
DATA = HERE / 'data'
V4DIR = AG / 'o_v4_revenue_patch_official'
E50F = Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')
END_DAY = pd.Timestamp('2026-10-12')

def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def grid():
    p = pd.read_csv(V4DIR / 'plan_V4.csv')
    g = list(pd.to_datetime(p['signal'])) + [END_DAY]
    return pd.DatetimeIndex(g)

GRID = grid()                      # 246 個點 → 245 個窗
WIN = GRID[:-1]                    # 窗標籤＝起點訊號日

def sample(nav, g=GRID):
    """日頻淨值在格點上取值（asof：格點當日或之前最後一筆）；淨值序列結束之後為 NaN。"""
    s = nav.dropna()
    if len(s) == 0: return pd.Series(np.nan, index=g)
    s = s[~s.index.duplicated(keep='last')].sort_index()
    v = s.reindex(s.index.union(g)).ffill().reindex(g)
    v[g < s.index[0]] = np.nan
    v[g > s.index[-1] + pd.Timedelta(days=7)] = np.nan   # 淨值結束超過一週 → 視為已停止
    return v

def win_ret(nav, g=GRID):
    v = sample(nav, g).to_numpy()
    r = v[1:] / v[:-1] - 1
    return pd.Series(r, index=g[:-1])

def e50_nav():
    e = pd.read_csv(E50F, parse_dates=['date']).set_index('date')['adj'].sort_index()
    return e

# 三個窗（窗起點訊號日）
WINDOWS = {
    'W_long': ('2006-04-01', '2026-10-12'),
    'W_19': ('2019-05-01', '2026-10-12'),
    'W_21': ('2021-03-01', '2026-10-12'),
}
# 子期間（只用於描述與挑選期探索）
SUBPER = {'SEL': ('2006-04-01', '2019-04-30'), 'VAL': ('2019-05-01', '2023-12-31'), 'HOLD': ('2024-01-01', '2026-10-12')}

def wmask(idx, a, b):
    return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(len(x))) if s > 0 else float('nan')

def canon_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

# ---------------- PREREG ----------------
RES_F = HERE / 'results.json'
def load_res():
    return json.loads(RES_F.read_text(encoding='utf-8')) if RES_F.exists() else {}

def check_prereg():
    """樣本外（任何 p 值）計算前的斷言：PREREG 存在、未被改、所有登記檔案（腳本與資料）的 sha256 與登記相同。"""
    res = load_res()
    assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：先跑 prereg.py'
    assert canon_sha(res['PREREG']) == res['PREREG_sha256'], 'PREREG 被改過'
    for rel, h in res['PREREG']['hashes'].items():
        p = Path(rel) if rel.startswith('/') else HERE / rel
        assert sha(p) == h, f'{rel} 與 PREREG 登記的 sha256 不符'
    return res

def save_key(key, obj):
    res = load_res()
    assert 'PREREG' in res
    res[key] = obj
    RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
