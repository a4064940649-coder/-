"""replicate_o2 共用：自寫事件表、異常報酬。主策略部分 import main_v2/v2_core（共用正式版規格，唯讀）。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parents[1]
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
EV = AG / 'o_index_deletion_flow' / 'events_v2.csv'
RD = AG / 'o_msci_inclusion_avoid' / 'review_dates.csv'

C = pd.read_parquet(P / 'close_raw.parquet'); RET = pd.read_parquet(P / 'ret.parquet'); V = pd.read_parquet(P / 'value.parquet')
SRC = pd.read_parquet(P / 'src.parquet')
IDX = C.index; COLS = C.columns; CI = {c: j for j, c in enumerate(COLS)}
Cn = C.to_numpy(); TR = np.isfinite(Cn)
R = np.clip(RET.to_numpy(), -0.5, 0.5)
R0 = np.where(TR & np.isfinite(R), R, 0.0)
MKT = np.nanmean(np.where(TR & np.isfinite(R), R, np.nan), axis=1)
MKT = np.nan_to_num(MKT); AR = R0 - MKT[:, None]
NOBS = TR.sum(1)
GOOD = NOBS >= 300
# 60 日均成交值（剔除 <300 檔的日子）
_g = np.where(GOOD)[0]
_v = pd.DataFrame(V.to_numpy()[_g]).rolling(60, min_periods=int(__import__("os").environ.get("MP","40"))).mean().to_numpy()
V60 = np.full(Cn.shape, np.nan); V60[_g] = _v
V60 = pd.DataFrame(V60).ffill(limit=1).to_numpy()
CF = C.ffill().to_numpy()

def tday_le(d):
    d = pd.Timestamp(d); i = IDX.searchsorted(d, side='right') - 1; return i
def tday_ge(d):
    d = pd.Timestamp(d); return IDX.searchsorted(d, side='left')

SRC3 = (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23'))
HD = ['0056', '00878', '00919', '00713', '00918', '00929', '00940']

def load_events():
    e = pd.read_csv(EV, dtype={'code': str, 'review': str})
    rd = pd.read_csv(RD, dtype={'review': str})
    ann = dict(zip(rd['review'], rd['ann_tw']))
    std_del = set(zip(e.loc[e.type == 'MSCI_STD_DEL', 'review'], e.loc[e.type == 'MSCI_STD_DEL', 'code']))
    rows = []
    for _, r in e.iterrows():
        t = r['type']
        if t == 'MSCI_STD_ADD': cls = 'STD'
        elif t == 'MSCI_SC_ADD':
            if (r['review'], r['code']) in std_del: continue
            cls = 'SC'
        elif t == 'TW50_ADD': cls = 'TW50'
        elif t.endswith('_ADD') and t[:-4] in HD: cls = 'HDETF'
        else: continue
        rows.append(dict(review=r['review'], cls=cls, code=r['code'], name=r['name'], eff=r['eff'], family=r['family'], idx=r['index']))
    d = pd.DataFrame(rows)
    pri = {'STD': 0, 'SC': 1, 'TW50': 2, 'HDETF': 3}
    d['pri'] = d['cls'].map(pri)
    # 同一審核同一檔只留一筆（STD 優先）。審核＝(家族/指數組, review)。MSCI 的 STD 與 SC 同屬一次審核
    d['grp'] = np.where(d['family'] == 'MSCI', 'MSCI', d['idx'])
    d = d.sort_values(['grp', 'review', 'code', 'pri']).drop_duplicates(['grp', 'review', 'code'])
    d['ann_tw'] = d['review'].map(ann).where(d['family'] == 'MSCI')
    d['Ei'] = [tday_le(x) for x in d['eff']]
    d['E'] = IDX[d['Ei']]
    d['cluster'] = d['cls'] + '|' + d['E'].astype(str)   # 按生效日分群
    return d.drop(columns='pri').reset_index(drop=True)

def car(j, a, b):
    """AR 加總，交易日位置 a..b（含）。"""
    if a < 0 or b >= len(IDX): return np.nan
    return float(AR[a:b + 1, j].sum())

def add_metrics(d):
    out = []
    for _, r in d.iterrows():
        j = CI.get(r['code']); Ei = r['Ei']
        o = dict(inpanel=j is not None)
        if j is not None:
            o['liq_ok'] = bool(np.isfinite(V60[Ei - 1, j]) and V60[Ei - 1, j] >= 1e7 and np.isfinite(CF[Ei - 1, j]) and CF[Ei - 1, j] >= 10 and TR[Ei, j])
            o['car15'] = car(j, Ei + 1, Ei + 5) if Ei + 5 < len(IDX) else np.nan
            o['arE'] = float(AR[Ei, j])
            if isinstance(r.get('ann_tw'), str):
                Ai = tday_ge(r['ann_tw'])
                o['Ai'] = Ai
                o['pre10'] = car(j, Ai - 10, Ai - 1); o['arA'] = float(AR[Ai, j])
                o['A_to_E'] = car(j, Ai, Ei); o['A1_to_Em1'] = car(j, Ai + 1, Ei - 1); o['A_to_Em1'] = car(j, Ai, Ei - 1); o['A1_to_E'] = car(j, Ai + 1, Ei)
        else:
            o['liq_ok'] = False
        out.append(o)
    return pd.concat([d, pd.DataFrame(out)], axis=1)

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return float('nan')
    s = x.std(ddof=1); return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def clus(df, col='car15'):
    x = df[np.isfinite(df[col])]
    g = x.groupby('cluster')[col].mean()
    return dict(n=int(len(x)), k=int(len(g)), mean=round(float(g.mean()) * 100, 2) if len(g) else None, t=round(tstat(g.values), 2),
                ev_mean=round(float(x[col].mean()) * 100, 2) if len(x) else None)
