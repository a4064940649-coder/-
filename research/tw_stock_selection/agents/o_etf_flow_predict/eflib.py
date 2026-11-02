"""o_etf_flow_predict 共用：面板、PIT 股數、臺灣50 成分、審核日期。只讀其他資料夾。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent
PANEL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
EX = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
_c = {}
def P(name):
    if name not in _c:
        _c[name] = pd.read_parquet(PANEL / f'{name}.parquet')
    return _c[name]
# 已知 PIT 股數錯誤（o9 題 4 指示修正）：4763 材料-KY 2025-05 面額 10→1 元，o8 股數在變更前已是變更後口徑（多 10 倍）。
PATCH_4763 = ('4763', pd.Timestamp('2025-05-02'), 0.1)
def shares_pit(lag=0, patch=False):
    k = f'sh{lag}_{patch}'
    if k not in _c:
        s = pd.read_parquet(AG / 'o_0050_tilt_pit/data/shares_pit.parquet')
        if patch:
            s = s.copy(); c, d, f = PATCH_4763
            s.loc[s.index < d, c] = s.loc[s.index < d, c] * f
        _c[k] = s.shift(lag) if lag else s
    return _c[k]
def reviews():
    r = pd.read_csv(AG / 'etf_rebalance/reviews.csv', dtype=str)
    return r
def events():
    return pd.read_csv(AG / 'etf_rebalance/events.csv', dtype={'code': str})
def tw50_members_before():
    d = json.load(open(AG / 'o_0050_tilt_pit/raw/agents/tw50_members.json'))
    steps = sorted(d['steps_backward'], key=lambda s: s['eff'])
    return d['current_2026-10-09'], steps
def tw50_members_at(date):
    """date（審核截止日）時的成分：最後一個 eff <= date 的變動之後的名單。"""
    cur, steps = tw50_members_before()
    date = pd.Timestamp(date)
    # 變動後名單：steps[i+1].members_before 或 current
    for i, s in enumerate(steps):
        if pd.Timestamp(s['eff']) > date:
            return set(s['members_before'])
    return set(cur)
def is_common(c):
    return len(c) == 4 and c.isdigit() and not c.startswith('0') and not c.startswith('91')
def tday_le(d):
    idx = P('close_raw').index
    d = pd.Timestamp(d)
    return idx[idx <= d][-1]
def tday_after(d, k=1):
    idx = P('close_raw').index
    d = pd.Timestamp(d)
    a = idx[idx > d]
    return a[k - 1] if len(a) >= k else None
def mcap_rank(date, lag=0, universe=None, patch=False):
    """date（交易日）收盤 × PIT 股數 的全市值名次（上市普通股，當日有收盤）。"""
    cl = P('close_raw').loc[date]
    sh = shares_pit(lag, patch)
    sh = sh.loc[sh.index[sh.index <= date][-1]] if (sh.index <= date).any() else None
    if sh is None: return None
    mc = (cl * sh.reindex(cl.index)).dropna()
    mc = mc[[c for c in mc.index if is_common(c)]]
    mc = mc[mc > 0]
    if universe is not None: mc = mc[mc.index.isin(universe)]
    return mc.rank(ascending=False, method='first').sort_values(), mc
