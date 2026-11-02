#!/usr/bin/env python3
"""replicate_o10 r0：獨立重建合併面板（上市 V7 凍結面板＋上櫃 tpex_main 面板＋三個 GitHub 封存補洞）。
沒有讀 o_tpex_pool_extension 研究員的任何程式。輸出：scratchpad/rep_o10/merged.npz 與 out/r0_panel.json。"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parents[1]
O7 = AG / 'o_v7_listing_exec_engine'; PNL = O7 / 'data'; RAW = O7 / 'raw'
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
TPP = SCR / 'tpex_panel'; OUTD = SCR / 'rep_o10'; OUTD.mkdir(exist_ok=True)
EXT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
(HERE / 'out').mkdir(exist_ok=True)
t0 = time.time()
KEYS = ['close_raw', 'open', 'value', 'ret', 'flag', 'high']
V7 = {k: pd.read_parquet(PNL / f'{k}.parquet') for k in KEYS}
IDX = V7['close_raw'].index
TPX = {k: pd.read_parquet(TPP / f'{k}.parquet').reindex(IDX) for k in KEYS}
tcols = list(TPX['close_raw'].columns); vcols = list(V7['close_raw'].columns)
trans = sorted(set(tcols) & set(vcols))
info = dict(n_v7=len(vcols), n_tpex=len(tcols), n_transfer=len(trans))

# ---------- 補洞（只補上櫃面板缺值格；只補上櫃期間） ----------
TC = TPX['close_raw'].to_numpy().copy()
TO = TPX['open'].to_numpy().copy(); TV = TPX['value'].to_numpy().copy(); TH = TPX['high'].to_numpy().copy()
TR_ = TPX['ret'].to_numpy().copy(); TF = TPX['flag'].to_numpy().copy()
TCI = {c: j for j, c in enumerate(tcols)}
DPOS = {d: i for i, d in enumerate(IDX)}
# 轉上市股：V7 面板第一個有價日之前才算上櫃
v7first = {c: V7['close_raw'][c].first_valid_index() for c in trans}
istp = np.ones(TC.shape, bool)
for c in trans:
    istp[IDX >= v7first[c], TCI[c]] = False
orig_has = np.isfinite(TC)
CHG = np.full(TC.shape, np.nan)   # 官方漲跌（補值格）
FILLED = np.zeros(TC.shape, np.int8)


def load_hschi():
    rows = []
    for p in sorted((RAW / 'hschi' / 'stock_dayK').glob('*.csv')):
        c = p.stem
        if c not in TCI or c == '6208': continue
        d = pd.read_csv(p, parse_dates=['date'])
        d = d[(d.date >= '2021-05-01') & (d.date <= '2023-09-30')]
        if len(d) == 0: continue
        rows.append(pd.DataFrame(dict(date=d.date, code=c, open=d.open, high=d['max'], close=d.close, value=d.Trading_money, chg=d.spread)))
    return pd.concat(rows)


def load_tasi():
    rows = []
    for p in sorted((RAW / 'tasi' / 'kbar').glob('kbar*.csv')):
        ds = p.stem[4:]
        if not ('2023-10-01' <= ds <= '2025-03-31'): continue
        try: d = pd.read_csv(p, dtype={'stoc': str})
        except Exception: continue
        if 'stoc' not in d.columns or len(d) == 0: continue
        d['stoc'] = d.stoc.astype(str).str.zfill(4)
        d = d[d.stoc.isin(TCI)]
        rows.append(pd.DataFrame(dict(date=pd.to_datetime(d['date']), code=d.stoc, open=d.ope, high=d['max'], close=d.clo,
                                      value=d.vol * 1000.0 * d.clo, chg=np.nan)))
    return pd.concat(rows)


def load_bacardi():
    rows = []
    for y in (2025,):
        d = pd.read_csv(RAW / 'bacardi' / f'prices_{y}.csv', dtype={'stock_id': str}, encoding='utf-8-sig', parse_dates=['date'])
        d = d[(d.market == 'TPEx') & (d.date >= '2025-04-01') & (d.date <= '2025-09-23') & d.stock_id.isin(TCI)]
        rows.append(pd.DataFrame(dict(date=d.date, code=d.stock_id, open=d.open, high=d.high, close=d.close, value=d.turnover, chg=d.change)))
    return pd.concat(rows)


src_stats = {}
for name, fn, sid in (('hschi', load_hschi, 1), ('tasi', load_tasi, 2), ('bacardi', load_bacardi, 3)):
    d = fn(); d = d[d.date.isin(DPOS)]
    d = d.drop_duplicates(['date', 'code'], keep='last')
    d = d[(d.close > 0) & np.isfinite(d.close)]
    ii = d.date.map(DPOS).to_numpy(); jj = d.code.map(TCI).to_numpy()
    tp = istp[ii, jj]; ii, jj, d = ii[tp], jj[tp], d[tp]
    ov = orig_has[ii, jj]
    pc = TC[ii[ov], jj[ov]]; sc = d.close.to_numpy()[ov]
    agree = float(np.mean(np.abs(sc - pc) < 0.005 + 1e-9))
    nw = ~np.isfinite(TC[ii, jj])
    a, b, dd = ii[nw], jj[nw], d[nw]
    TC[a, b] = dd.close.to_numpy(); TO[a, b] = dd.open.to_numpy(); TH[a, b] = dd.high.to_numpy(); TV[a, b] = dd.value.to_numpy()
    CHG[a, b] = dd.chg.to_numpy(); FILLED[a, b] = sid; TF[a, b] = 0
    src_stats[name] = dict(overlap=int(ov.sum()), agree=agree, filled=int(nw.sum()))
info['fill'] = src_stats
print(src_stats, round(time.time() - t0), flush=True)

# ---------- 報酬：補值格與跨缺口第一天 ----------
EX = pd.read_csv(EXT / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
EXR = {(DPOS[d], TCI[c]): r for d, c, r in zip(EX.date, EX.code, EX.ref_price) if d in DPOS and c in TCI and r > 0}
CFw = pd.DataFrame(TC).ffill().shift(1).to_numpy()            # 補值後前收
origCF = pd.DataFrame(np.where(orig_has, TC, np.nan)).ffill().shift(1).to_numpy()
# 前一個有價日是否為補值格
lastfill = pd.DataFrame(np.where(np.isfinite(TC), (FILLED > 0).astype(float), np.nan)).ffill().shift(1).to_numpy()
need_f = FILLED > 0
need_x = orig_has & (lastfill == 1)
n_clip = 0
for kind, M in (('fill', need_f), ('cross', need_x)):
    ii, jj = np.where(M)
    for i, j in zip(ii, jj):
        c = TC[i, j]; pcl = CFw[i, j]
        if (i, j) in EXR: r = c / EXR[(i, j)] - 1
        elif kind == 'fill' and np.isfinite(CHG[i, j]) and c - CHG[i, j] > 0: r = c / (c - CHG[i, j]) - 1
        elif np.isfinite(pcl) and pcl > 0: r = c / pcl - 1
        else: r = np.nan
        lim = 0.07 if IDX[i] < pd.Timestamp('2015-06-01') else 0.10
        if np.isfinite(r) and abs(r) > lim + 5e-4:
            r = 0.0; TF[i, j] = 3; n_clip += 1
        TR_[i, j] = r
info['n_fill_ret'] = int(need_f.sum()); info['n_crossgap_recalc'] = int(need_x.sum()); info['n_clip'] = n_clip

# ---------- 洞月份（gap_map 定義） ----------
TPm = istp & True
has = np.isfinite(TC) & istp
mon = IDX.to_period('M')
cov = {}
anyb = pd.DataFrame(has).groupby(mon).any()
pm = sorted(set(mon))
cumb = anyb.cumsum(); rcum = anyb[::-1].cumsum()[::-1]
for k, p in enumerate(pm):
    if k == 0 or k == len(pm) - 1: continue
    alive = (cumb.loc[pm[k - 1]].to_numpy() > 0) & (rcum.loc[pm[k + 1]].to_numpy() > 0)
    m = mon == p
    med = np.median(has[m].sum(1))
    cov[str(p)] = float(med / max(alive.sum(), 1))
holes = [p for p, v in cov.items() if v < 0.85 and p >= '2007-06']
low90 = [p for p, v in cov.items() if v < 0.90 and p >= '2007-06']
info['holes_lt85'] = holes; info['months_lt90'] = low90
print('holes', holes, flush=True)

# ---------- 合併 ----------
cols = sorted(set(vcols) | set(tcols))
CI = {c: j for j, c in enumerate(cols)}
N = len(cols); T = len(IDX)
def merged(vk, tarr):
    A = V7[vk].reindex(columns=cols).to_numpy().astype(float)
    for c in tcols:
        j = CI[c]; jt = TCI[c]
        m = istp[:, jt] & np.isfinite(TC[:, jt]) & ~np.isfinite(A[:, j])
        A[m, j] = tarr[m, jt]
    return A
C = merged('close_raw', TC); O = merged('open', TO); V = merged('value', TV); R = merged('ret', TR_); F = merged('flag', TF); H = merged('high', TH)
TPX_MASK = np.zeros((T, N), bool)
for c in tcols:
    TPX_MASK[:, CI[c]] = istp[:, TCI[c]]
FILL = np.zeros((T, N), np.int8)
for c in tcols:
    FILL[:, CI[c]] = np.where(istp[:, TCI[c]], FILLED[:, TCI[c]], 0)
# 轉上市首日：前一日是上櫃價 → 用前收接續（V7 首日報酬若缺）
n_splice = 0
for c in trans:
    j = CI[c]; i = IDX.get_loc(v7first[c])
    if not np.isfinite(R[i, j]):
        w = np.where(np.isfinite(C[:i, j]))[0]
        if len(w):
            r = C[i, j] / C[w[-1], j] - 1
            R[i, j] = r if abs(r) < 0.1005 else 0.0; n_splice += 1
info['n_splice'] = n_splice
# 下櫃股（非轉上市）：最後有價日在面板終點 30 天之前
lt = np.where(np.isfinite(C).any(0), T - 1 - np.argmax(np.isfinite(C)[::-1], 0), -1)
tp_only = [c for c in tcols if c not in set(vcols)]
delist_tp = [c for c in tp_only if lt[CI[c]] >= 0 and IDX[lt[CI[c]]] < IDX[-1] - pd.Timedelta(days=30)]
info['n_delist_tpex'] = len(delist_tp)
np.savez_compressed(OUTD / 'merged.npz', IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(cols, dtype=str), C=C, O=O, V=V, R=R, F=F, H=H,
                    TPX=TPX_MASK, FILL=FILL, DELIST_TP=np.array(delist_tp, dtype=str), LOW90=np.array(low90, dtype=str))
info['cov'] = cov
json.dump(info, open(HERE / 'out' / 'r0_panel.json', 'w'), ensure_ascii=False, indent=1)
print({k: v for k, v in info.items() if k != 'cov'}, round(time.time() - t0), flush=True)
