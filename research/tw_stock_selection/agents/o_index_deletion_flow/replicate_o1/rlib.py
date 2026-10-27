"""獨立複製（replicate_o1）共用函式：只讀面板、凍結事件表，不 import 研究員程式。"""
import json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
E50F = Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')
EVF = TOP / 'events_v2.csv'
FROZEN_SHA = 'b19b78f2d36d2a7de29e78d99aa8ced1fed24fc5eae7c309b72cf6a6aa705759'

SRC3 = (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23'))
SEL = (pd.Timestamp('2008-01-01'), pd.Timestamp('2018-12-31'))
VAL = (pd.Timestamp('2019-01-01'), pd.Timestamp('2023-12-31'))
HOLD = (pd.Timestamp('2025-09-24'), pd.Timestamp('2026-10-31'))
KCLS = {'MSCI_STD_DEL': 5e8, 'TW50_DEL': 3e8, 'MSCI_SC_DEL': 0.5e8}
HD_ETF = ['0056', '00878', '00919', '00713', '00918', '00929', '00940']
for _e in HD_ETF: KCLS[f'{_e}_DEL'] = 3e8
CUT = 2.7478
COST, STRESS = 0.005, 0.01
H = 20

_cache = {}
def panel():
    if 'p' in _cache: return _cache['p']
    C = pd.read_parquet(P / 'close_raw.parquet')
    n = C.notna().sum(axis=1)
    good = n >= 300
    d = {}
    for k in ['adj', 'close_raw', 'open', 'high', 'value', 'ret', 'src', 'flag']:
        x = pd.read_parquet(P / f'{k}.parquet')
        d[k] = x[good.reindex(x.index).fillna(False).to_numpy()]
    d['idx'] = d['close_raw'].index
    d['adjf'] = d['adj'].ffill()
    d['traded'] = d['close_raw'].notna() & (d['value'].fillna(0) > 0)
    v = d['value']
    d['adv20'] = v.rolling(20, min_periods=15).mean()
    d['adv60'] = v.rolling(60, min_periods=40).mean()
    d['volmed'] = v.shift(6).rolling(20, min_periods=15).median()   # E-25..E-6 中位數
    e50 = pd.read_csv(E50F, parse_dates=['date']).set_index('date')['adj'].sort_index()
    d['e50'] = e50.reindex(d['idx'].union(e50.index)).ffill().reindex(d['idx'])
    # 等權日報酬（當日有成交者）
    d['ewret'] = d['ret'].where(d['traded']).mean(axis=1)
    _cache['p'] = d
    return d

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_events():
    assert sha(EVF) == FROZEN_SHA, 'events_v2.csv 被改過'
    e = pd.read_csv(EVF, dtype={'code': str}, parse_dates=['eff'])
    return e

def ew_bh(t0, t1):
    """t0 收盤 → t1 收盤，全體上市股（t0 有成交）買進持有等權報酬。"""
    d = panel()
    A = d['adjf']
    a0 = d['adj'].iloc[t0]
    ok = a0.notna() & d['traded'].iloc[t0]
    r = A.iloc[t1][ok] / a0[ok] - 1
    return float(r.mean())

def stk_bh(code, t0, t1):
    d = panel()
    a0 = d['adj'][code].iloc[t0]
    a1 = d['adjf'][code].iloc[t1]
    return float(a1 / a0 - 1)

def tstat(x):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) < 2: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def cl_t(df, col, key='month'):
    """事件月分群 t：每月平均 → 月份 t。"""
    m = df.groupby(key)[col].mean()
    return tstat(m), int(len(m))

def trim_mean(x, q=0.05):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) == 0: return float('nan')
    return float(x[x <= np.quantile(x, 1 - q)].mean())   # 剔除高於 95 分位數者

def summ(df, col):
    if len(df) == 0: return {'n': 0}
    t, nm = cl_t(df, col)
    return {'n': int(len(df)), 'n_months': nm, 'mean': round(df[col].mean() * 100, 2), 'median': round(df[col].median() * 100, 2),
            't_month': round(t, 2), 'trim5': round(trim_mean(df[col]) * 100, 2), 'win': round(float((df[col] > 0).mean()), 3)}

def hl(df, col):
    """高減低：簡單差、同事件月配對差與 t。"""
    hi, lo = df[df.high], df[~df.high]
    mh, ml = hi.groupby('month')[col].mean(), lo.groupby('month')[col].mean()
    j = mh.index.intersection(ml.index)
    dd = mh[j] - ml[j]
    return {'simple': round((hi[col].mean() - lo[col].mean()) * 100, 2), 'paired': round(dd.mean() * 100, 2) if len(dd) else None,
            't_paired': round(tstat(dd), 2) if len(dd) > 1 else None, 'n_months_paired': int(len(dd))}

def dump(obj, name):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))

def load(name):
    return json.loads((HERE / name).read_text())

def event_frame():
    """每筆事件：可交易篩選旗標、pressure、各窗口報酬（E 收盤起算；異常＝個股 − 全體上市等權）。"""
    f = HERE / 'event_frame.pkl'
    if f.exists(): return pd.read_pickle(f)
    d = panel(); idx = d['idx']; T = len(idx)
    e = pd.read_pickle(HERE / 'events_r.pkl')
    C = d['close_raw']; Cf = C.ffill()
    rows = []
    ewc = {}
    def EW(a, b):
        if (a, b) not in ewc: ewc[(a, b)] = ew_bh(a, b)
        return ewc[(a, b)]
    for r in e.itertuples():
        o = {'i': r.Index}
        t = int(r.t)
        o['ok_panel'] = bool(r.in_panel) and t >= 61
        if not o['ok_panel']:
            rows.append(o); continue
        c = r.code
        o['adv60'] = d['adv60'][c].iloc[t - 1]; o['adv20'] = d['adv20'][c].iloc[t - 1]
        o['pc'] = Cf[c].iloc[t - 1]; o['cE'] = C[c].iloc[t]
        o['tradedE'] = bool(d['traded'][c].iloc[t])
        lim = 0.065 if r.eff < pd.Timestamp('2015-06-01') else 0.095
        o['limup'] = bool(np.isfinite(o['cE']) and o['cE'] >= o['pc'] * (1 + lim) - 1e-9)
        o['liq'] = bool(np.isfinite(o['adv60']) and o['adv60'] >= 1e7 and np.isfinite(o['pc']) and o['pc'] >= 10)
        o['vr'] = d['value'][c].iloc[t] / d['volmed'][c].iloc[t] if d['volmed'][c].iloc[t] > 0 else np.nan
        o['K'] = KCLS.get(r.type, np.nan)
        o['pressure'] = o['K'] / o['adv20'] if np.isfinite(o['K']) and o['adv20'] > 0 else np.nan
        if o['tradedE']:
            rt = d['ret'][c]
            o['abE'] = rt.iloc[t] - d['ewret'].iloc[t]
            if t + 1 < T: o['abE1'] = rt.iloc[t + 1] - d['ewret'].iloc[t + 1] if d['traded'][c].iloc[t + 1] else np.nan
            o['pre10'] = stk_bh(c, t - 11, t - 1) - EW(t - 11, t - 1) if np.isfinite(d['adj'][c].iloc[t - 11]) else np.nan
            for h in (1, 5, 10, 20):
                if t + h < T:
                    o[f'g{h}'] = stk_bh(c, t, t + h); o[f'ew{h}'] = EW(t, t + h)
                    o[f'car{h}'] = o[f'g{h}'] - o[f'ew{h}']
            if t + 20 < T:
                o['car1_20'] = o['car20'] - 0  # E 收盤 → E+20 收盤（＝E+1～E+20 合計）
                o['car2_20'] = (1 + o['g20']) / (1 + o['g1']) - 1 - ((1 + o['ew20']) / (1 + o['ew1']) - 1)
            # 逐日異常加總（CAR）：E+1..E+5、E+1..E+20、E-10..E-1
            ar = (d['ret'][c] - d['ewret']).to_numpy()
            if t + 20 < T:
                o['sc5'] = np.nansum(ar[t + 1:t + 6]); o['sc20'] = np.nansum(ar[t + 1:t + 21])
            o['sp10'] = np.nansum(ar[t - 10:t])
            # 0050 同窗
            if t + 20 < T: o['r50_20'] = d['e50'].iloc[t + 20] / d['e50'].iloc[t] - 1
        rows.append(o)
    F = pd.DataFrame(rows).set_index('i')
    x = e.join(F)
    x['month'] = x.eff.dt.to_period('M')
    x['is_del'] = x.type.str.endswith('_DEL')
    x['cls'] = np.where(x.type.str.match(r'^00\d+_'), 'HD_' + x.type.str.split('_').str[-1], x.type)
    for k, dv in (('ok_panel', False), ('liq', False), ('tradedE', False), ('limup', True)):
        x[k] = x[k].astype(object).where(x[k].notna(), dv).astype(bool)
    x['src3_period'] = x['src3_period'].astype(bool)
    x['explorable'] = x.ok_panel & x.liq & x.tradedE & ~x.src3_period & x.car20.notna()   # 事件研究用（不排除漲停）
    x['tradable'] = x.explorable & ~x.limup                                                # 交易用（E 收盤漲停買不到）
    x['win'] = np.select([(x.eff >= SEL[0]) & (x.eff <= SEL[1]), (x.eff >= VAL[0]) & (x.eff <= VAL[1]), (x.eff >= HOLD[0]) & (x.eff <= HOLD[1])],
                         ['select', 'validate', 'holdout'], 'other')
    x['high'] = x.pressure >= CUT
    x['net20'] = x.car20 - COST; x['net20s'] = x.car20 - STRESS
    x['net20_50'] = x.g20 - COST - x.r50_20
    x.to_pickle(f)
    return x
