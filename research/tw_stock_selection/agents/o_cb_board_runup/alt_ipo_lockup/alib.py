# -*- coding: utf-8 -*-
"""替代題共用：主策略 v2（main_v2/v2_core，只讀 import）、上市年資、持有期報酬、避開版本的換股計畫。"""
import os, sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
if 'o3_core' in sys.modules or os.environ.get('ALT_V3') == '1':
    sys.path.insert(0, str(AG / 'o_infra_v3_rebuild'))
    import o3_core as O
    sys.path.pop(0)
    V, K = O.V, O.K
else:
    O = None
    sys.path.insert(0, str(AG / 'main_v2'))
    import v2_core as V
    sys.path.pop(0)
    K = V.K
IDX, COLS, N, CI = K.IDX, K.COLS, K.N, K.CI

SEL_END = pd.Timestamp('2019-04-30')
VAL_START = pd.Timestamp('2019-05-01')
P1_START = pd.Timestamp('2007-01-01')   # 年資分組從 2007 起（2004-01 以前上市的老股到 2007 年必已滿 3 年）

LS = pd.read_csv(HERE / 'listings.csv', dtype={'code': str}, parse_dates=['first', 'last'])
LS = LS.set_index('code').reindex(COLS)
KIND = LS['kind'].fillna('old').to_numpy()
FIRST = LS['first'].to_numpy(dtype='datetime64[ns]')
NEWMASK = np.isin(KIND, ['ipo', 'transfer'])
IPOMASK = KIND == 'ipo'
TRMASK = KIND == 'transfer'
EXCL_KIND = np.isin(KIND, ['reorg', 'dr'])    # 控股轉換、DR：不算新上市，也不算老股（分析時排除）
_ipo_year = LS[LS.kind == 'ipo']['first'].dt.year.value_counts().sort_index()
HOT_MED = float(_ipo_year.loc[2004:2018].median())
HOTYEAR = {int(y): bool(c >= HOT_MED) for y, c in _ipo_year.items()}
HOT = np.array([HOTYEAR.get(pd.Timestamp(f).year, False) if NEWMASK[j] else False for j, f in enumerate(FIRST)])

def age_years(d):
    """訊號日 d 的上市年資（年）；老股（2004-06 以前已在面板）給 99。"""
    a = (np.datetime64(pd.Timestamp(d)) - FIRST) / np.timedelta64(1, 'D') / 365.25
    return np.where(NEWMASK | EXCL_KIND, a, 99.0)

ELEC_KEYS = ('電子', '半導體', '光電', '通信網路', '電腦及週邊', '資訊服務')
ELEC = np.array([any(k in str(i) for k in ELEC_KEYS) for i in V.IND])

def half_year_anniv_in(j, t_from, t_to, months=(12, 18, 24, 30, 36)):
    """上市後第 12/18/24/30/36 個月週年是否落在 (IDX[t_from], IDX[t_to]]（集保分批領回的候選日；各年度規則未全查證，用半年週年當代理）。"""
    if not NEWMASK[j]: return False
    f = pd.Timestamp(FIRST[j]); a, b = IDX[t_from], IDX[min(t_to, len(IDX) - 1)]
    for m in months:
        x = f + pd.DateOffset(months=m)
        if a < x <= b: return True
    return False

def per_ret(t, t2, pk):
    """t 開盤買、t2 開盤賣（含息；t2 無成交則到 t2 收盤）。"""
    pk = np.asarray(pk, int)
    r = K.Cn[t, pk] / K.On[t, pk]
    if t2 > t: r = r * np.prod(K.GROW[t + 1:t2 + 1, pk], axis=0)
    tr = K.TRADED[t2, pk] & np.isfinite(K.On[t2, pk])
    r = np.where(tr, r * K.On[t2, pk] / np.where(tr, K.Cn[t2, pk], 1.0), r)
    return r - 1.0

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def msum(x):
    x = pd.Series(x).dropna()
    return dict(mean_pp=float(x.mean() * 100) if len(x) else None, sd_pp=float(x.std(ddof=1) * 100) if len(x) > 1 else None, n=int(len(x)), t=tstat(x))

def strategy(exfin=True, S_cls=None, Y=None):
    YL, Y5, _ = V.load_revenue() if Y is None else Y
    cls = S_cls or V.V2
    return cls('gapfix', YL, Y5, exfin=exfin)

def plan_avoid(S, n, avoid_fn=None, sell_held=False):
    """V2.plan 加避開名單：avoid_fn(d, t, t_next_guess) -> bool 陣列（True＝本期不買）。被避開者由下一名遞補。
    sell_held=False：已持有的被避開股如果仍在前 n 名就續抱（不強迫賣出）。回傳 (plan, skipped[(d, j)])。"""
    out = []; held = set(); nblk = 0; skipped = []
    sd = [d for d in S.sig_days]
    for i, d in enumerate(sd):
        cs = S.cross_section(d)
        el = cs['el']
        if el.sum() < 50: continue
        e = cs['e']; gu = K.gap_up_open(e)
        e2 = IDX.get_loc(sd[i + 1]) + 1 if i + 1 < len(sd) else IDX.get_loc(S.end_day)
        av = avoid_fn(d, e, e2) if avoid_fn is not None else np.zeros(N, bool)
        cand = S.order(cs, 'official')[:int(el.sum())]
        pick = []
        for j in cand:
            if av[j] and (sell_held or j not in held):
                skipped.append((d, j)); continue
            if gu[j] and j not in held: nblk += 1; continue
            pick.append(j)
            if len(pick) == n: break
        out.append((e, np.array(pick, int), np.where(el)[0], d))
        held = set(pick)
    S.nblk = nblk
    return out, skipped

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def conv(o):
    if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (pd.Timestamp,)): return str(o.date())
    if isinstance(o, float): return None if not math.isfinite(o) else round(o, 6)
    if isinstance(o, (pd.Period,)): return str(o)
    return str(o)

def dump(obj, name):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding='utf-8')
