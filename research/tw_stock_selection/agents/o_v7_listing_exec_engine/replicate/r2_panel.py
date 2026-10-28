#!/usr/bin/env python3
"""獨立複製（V7 面板）：以 o_v6 複製員的 rep_build.py 為底（不讀研究員的 s2_build7.py／srcload7.py／data/src_*.parquet），依本題 frozen_spec 改三處：
上市身分用複製員自建的 replicate/data/listed_rep.parquet；颱風停市日（不在面板日期）事件順延到下一個面板日期；窗後第一筆若 V4 已標 flag 2 保留原 ret。
原說明：只依 frozen_spec.txt，從 raw/ 原始 CSV 自己建 V6 面板。
寫法刻意不同：長表＋優先序排序去重、groupby 前收、事件表 merge；輸出 replicate/data/{close_raw,open,high,low,value,ret,flag,src,adj}.parquet。"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent; TOP = H.parent; RAW = TOP / 'raw'; OUT = H / 'data'; OUT.mkdir(exist_ok=True)
PNL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = RAW / 'screener' / 'extras'
WA, WB = pd.Timestamp('2024-01-02'), pd.Timestamp('2025-09-23')
V4 = {k: pd.read_parquet(PNL / f'{k}.parquet') for k in ('adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src')}
COLS = V4['close_raw'].columns; CS = set(COLS)
tx = pd.to_datetime(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv')['date'])
IDX = V4['close_raw'].index.union(pd.DatetimeIndex(sorted(set(tx[(tx >= WA) & (tx <= WB)]))))
POS = pd.Series(np.arange(len(IDX)), index=IDX)

def keep(d):
    d = d[d.code.isin(CS) & (d.date >= WA) & (d.date <= WB)]
    for c in ('o', 'h', 'l', 'c', 'v'): d[c] = pd.to_numeric(d[c], errors='coerce')
    return d[(d.v > 0) & (d.c > 0) & (d.o > 0)]

# ---- 原始來源（自己解析）----
rows = []
for f in glob.glob(str(RAW / 'hschi' / 'stock_dayK' / '*.csv')):
    x = pd.read_csv(f, dtype={'stock_id': str})
    rows.append(pd.DataFrame({'date': pd.to_datetime(x['date']), 'code': x['stock_id'], 'o': x['open'], 'h': x['max'], 'l': x['min'], 'c': x['close'],
                              'v': x['Trading_Volume'], 'val': x['Trading_money']}))
hs = keep(pd.concat(rows)); hs['p'] = 1
x = pd.read_csv(RAW / 'bacardi' / 'prices_2025.csv', dtype={'stock_id': str}, encoding='utf-8-sig'); x = x[x.market == 'TWSE']
bc = keep(pd.DataFrame({'date': pd.to_datetime(x['date']), 'code': x['stock_id'], 'o': x['open'], 'h': x['high'], 'l': x['low'], 'c': x['close'],
                        'v': x['volume'], 'val': x['turnover']})); bc['p'] = 2
rows = []
for f in glob.glob(str(RAW / 'leochen1_stock_v' / 'data' / 'history' / '*.csv')):
    x = pd.read_csv(f, encoding='utf-8-sig'); x.columns = ['date', 'o', 'h', 'l', 'c', 'v']
    x['code'] = Path(f).stem; x['date'] = pd.to_datetime(x['date']); x['val'] = np.nan; rows.append(x)
lc = keep(pd.concat(rows)); lc['p'] = 3
rows = []
for f in sorted(glob.glob(str(RAW / 'tasi' / 'kbar' / 'kbar*.csv'))):
    with open(f, 'rb') as fh: hd = fh.readline().strip()
    if hd != b'"stoc","date","max","min","ope","clo","vol"': continue
    x = pd.read_csv(f, dtype={'stoc': str})
    fd = pd.Timestamp(Path(f).name[4:14]); x['date'] = pd.to_datetime(x['date']); x = x[x['date'] == fd]
    rows.append(pd.DataFrame({'date': x['date'], 'code': x['stoc'], 'o': x['ope'], 'h': x['max'], 'l': x['min'], 'c': x['clo'],
                              'v': pd.to_numeric(x['vol'], errors='coerce') * 1000, 'val': np.nan}))
ta = keep(pd.concat(rows)); ta['p'] = 4
for d in (hs, bc, lc, ta): d.drop_duplicates(['date', 'code'], keep='last', inplace=True)
# ---- 上市身分（複製員自建上市矩陣）----
v4w = V4['close_raw'].loc[WA:WB].stack().dropna().reset_index(); v4w.columns = ['date', 'code', 'cv4']
LR = pd.read_parquet(H / 'data' / 'listed_rep.parquet')
lst_st = LR.loc[WA:WB].stack(); mem = set(lst_st[lst_st].index)
allc = pd.concat([hs, bc, lc, ta], ignore_index=True)
inm = np.array([(d, c) in mem for d, c in zip(allc.date, allc.code)])
v4w = v4w[[(d, c) in mem for d, c in zip(v4w.date, v4w.code)]]
allc = allc[inm].sort_values(['date', 'code', 'p'])
best = allc.drop_duplicates(['date', 'code'], keep='first').copy()
# 成交值：hschi＞bacardi（收盤要一致），否則 (o+h+l+c)/4×量
for src_df in (hs, bc):
    m = best.merge(src_df[['date', 'code', 'c', 'val']], on=['date', 'code'], how='left', suffixes=('', '_s'))
    use = m['val'].isna() & m['val_s'].notna() & (np.abs(m['c_s'] - m['c']) < 1e-6)
    best['val'] = np.where(use.to_numpy(), m['val_s'].to_numpy(), best['val'].to_numpy())
est = (best.o + best.h + best.l + best.c) / 4 * best.v
best['val'] = best['val'].where(best['val'].notna(), est)
best['src'] = best.p.map({1: 5, 2: 7, 3: 7, 4: 6})
# 備援：V4 src 3
have = set(zip(best.date, best.code))
fb = v4w[[(d, c) not in have for d, c in zip(v4w.date, v4w.code)]].copy()
fbx = pd.DataFrame({'date': fb.date, 'code': fb.code, 'c': fb.cv4,
                    'o': [V4['open'].at[d, c] for d, c in zip(fb.date, fb.code)], 'h': [V4['high'].at[d, c] for d, c in zip(fb.date, fb.code)],
                    'l': np.nan, 'val': [V4['value'].at[d, c] for d, c in zip(fb.date, fb.code)], 'src': 3})
W = pd.concat([best[['date', 'code', 'o', 'h', 'l', 'c', 'val', 'src']], fbx], ignore_index=True)
print('window cells', len(W), 'fallback', len(fbx))
# ---- 寬表 ----
win = (IDX >= WA) & (IDX <= WB)
F = {k: V4[k].reindex(IDX) for k in ('close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src', 'adj')}
F['low'] = pd.DataFrame(np.nan, index=IDX, columns=COLS)
for k, col in (('close_raw', 'c'), ('open', 'o'), ('high', 'h'), ('low', 'l'), ('value', 'val'), ('src', 'src')):
    w = W.pivot(index='date', columns='code', values=col).reindex(index=IDX[win], columns=COLS)
    F[k].loc[IDX[win]] = w.to_numpy()
# ---- 報酬：長表（全歷史收盤）→ 每檔前一筆 ----
cl = F['close_raw'].stack().dropna().reset_index(); cl.columns = ['date', 'code', 'c']
cl['pos'] = POS.reindex(cl.date).to_numpy()
cl = cl.sort_values(['code', 'pos'])
cl['prev_c'] = cl.groupby('code')['c'].shift(1); cl['prev_pos'] = cl.groupby('code')['pos'].shift(1)
cl['nprior'] = cl.groupby('code').cumcount()
tgt = cl[(cl.date >= WA) & (cl.date <= WB)].copy()
postf = cl[cl.date > WB].groupby('code').head(1).copy()
# 事件（上市）
ex = pd.read_csv(X / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); ex = ex[ex.market == 'TWSE'][['date', 'code', 'prev_close', 'ref_price']]
ca = pd.read_csv(X / 'corp_actions.csv', dtype={'code': str}, parse_dates=['date']); ca = ca[ca.market == 'TWSE'][['date', 'code', 'prev_close', 'ref_price']]
ca['prev_close'] = np.nan
ev = pd.concat([ex.assign(kind='exdiv'), ca.assign(kind='corp')]).drop_duplicates(['date', 'code'], keep='last')
ev = ev[ev.code.isin(CS)].copy()
not_in = ~ev.date.isin(IDX)
ev.loc[not_in, 'date'] = [IDX[IDX.searchsorted(d)] if IDX.searchsorted(d) < len(IDX) else pd.NaT for d in ev.loc[not_in, 'date']]   # 颱風停市日 → 下一個面板日期
ev = ev.dropna(subset=['date']).drop_duplicates(['date', 'code'], keep='last')
ev['pos'] = POS.reindex(ev.date).to_numpy()
# 對齊：事件日沒成交 → 5 日內下一個成交日；表列前收≠前一筆收盤但＝當日收盤 → 下一個成交日
trd = {c: g['pos'].to_numpy() for c, g in cl.groupby('code')}
cpos = {(c, p): v for c, p, v in zip(cl.code, cl.pos, cl.c)}
newpos = []
for c, p, pc in zip(ev.code, ev.pos, ev.prev_close):
    tp = trd.get(c, np.array([], int))
    if (c, p) not in cpos:
        nx = tp[(tp > p) & (tp <= p + 5)]; newpos.append(nx[0] if len(nx) else -1); continue
    k = np.searchsorted(tp, p)
    prevc = cpos[(c, tp[k - 1])] if k > 0 else np.nan
    if np.isfinite(pc) and abs(pc - prevc) > 1e-6 and abs(pc - cpos[(c, p)]) < 1e-6:
        nx = tp[(tp > p) & (tp <= p + 5)]; newpos.append(nx[0] if len(nx) else -1)
    else:
        newpos.append(p)
ev['pos'] = newpos; ev = ev[ev.pos >= 0]
tgt = tgt.merge(ev[['code', 'pos', 'ref_price']], on=['code', 'pos'], how='left')
lim = 0.105
r_plain = tgt.c / tgt.prev_c - 1; r_ev = tgt.c / tgt.ref_price - 1
has_ev = tgt.ref_price.notna()
reject = has_ev & (tgt.nprior > 5) & (r_ev.abs() > lim) & (r_plain.abs() <= lim)
use_ev = has_ev & ~reject
longgap = ~use_ev & tgt.prev_pos.notna() & ((tgt.pos - tgt.prev_pos) > 20)
ret = np.where(use_ev, r_ev, r_plain); flag = np.where(use_ev, 2, 0)
ret = np.where(longgap & (np.abs(r_plain) > lim), np.nan, ret); flag = np.where(longgap, 4, flag)
f5 = (flag == 0) & (tgt.nprior > 5) & (np.abs(ret) > lim)
ret = np.where(f5, np.nan, ret); flag = np.where(f5, 5, flag)
first = tgt.prev_c.isna(); ret = np.where(first, np.nan, ret); flag = np.where(first, 0, flag)
tgt['ret'] = ret; tgt['flag'] = flag
postf['ret'] = postf.c / postf.prev_c - 1; postf['flag'] = 0          # 窗後第一筆：依新前收重算
f4 = np.array([V4['flag'].at[d, c] if d in V4['flag'].index else np.nan for d, c in zip(postf.date, postf.code)])
postf = postf[f4 != 2]                                                    # V4 已標 flag 2（除權息）者保留原 ret
upd = pd.concat([tgt[['date', 'code', 'ret', 'flag']], postf[['date', 'code', 'ret', 'flag']]])
R = F['ret'].copy(); FL = F['flag'].copy()
R.loc[IDX[win]] = np.nan; FL.loc[IDX[win]] = np.nan
Ru = upd.pivot(index='date', columns='code', values='ret'); Fu = upd.pivot(index='date', columns='code', values='flag')
for d in Ru.index:
    cs = Ru.columns[Ru.loc[d].notna() | Fu.loc[d].notna()]
    R.loc[d, cs] = Ru.loc[d, cs].to_numpy(); FL.loc[d, cs] = Fu.loc[d, cs].to_numpy()
F['ret'], F['flag'] = R, FL
# ---- adj ----
t0 = int(np.argmax(win))
cn = F['close_raw'].to_numpy(); rn = F['ret'].to_numpy(); A = F['adj'].to_numpy().copy()
tr = np.isfinite(cn)
g = np.where(tr, 1 + np.nan_to_num(rn), 1.0)
A[t0:] = np.nan
for j in range(len(COLS)):
    prior = np.where(tr[:t0, j])[0]
    base = V4['adj'].iloc[:, j].reindex(IDX).to_numpy()[prior[-1]] if len(prior) else None
    rows_ = np.where(tr[t0:, j])[0] + t0
    if len(rows_) == 0: continue
    if base is None:
        vals = np.cumprod(g[rows_, j]) / g[rows_[0], j]
    else:
        vals = base * np.cumprod(g[rows_, j])
    A[rows_, j] = vals
F['adj'] = pd.DataFrame(A, index=IDX, columns=COLS)
for k, v in F.items(): v.astype(float).to_parquet(OUT / f'{k}.parquet')
print('written', {k: v.shape for k, v in F.items()})
