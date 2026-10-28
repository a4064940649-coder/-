#!/usr/bin/env python3
"""o_open_impact_calib 校準資料的共用前處理（只看執行量與價格路徑，不看任何策略報酬）。
  stockdays()：前 300 池（vrank_{t-1} ≤ 300）× Yahoo 5 分 K × 官方日 K 的股日表（每股每日一列）與 bar 表（09:05～09:55 共 11 根）。
  mis()：vol_curve（2026-10，每 3 分鐘累積量，MIS）× 官方日 K：開盤集合競價量真值代理、收盤集合競價量、09:00～09:05 連續量。
  oddlot()：oddlot_hist（2026-07-13 起，盤中零股收盤時點的價、量、bid、ask）× 官方日 K。
期間常數：CAL＝2024-02-15～2025-06-30（校準）、VAL＝2025-07-01～2026-04-01（驗證），事前固定。
定義：
  σ＝sig60_{t-1}（60 個 GOOD 日日報酬標準差）；V60sh＝V60_{t-1} ÷ 前收（股）；dec＝ceil(vrank_{t-1}/30)（1～10，1＝最大）。
  bar 報酬 r_b＝ln(C_b ÷ C_{b−5分})，基準 C_09:00 為 09:00 K 棒收盤；q_b＝該 K 棒 Yahoo 量（股）。
  VWAP_0905＝09:05～09:25 五根 K 棒（09:05～09:30）的 (H+L+C)/3 量加權；開盤溢價 prem＝ln(官方開盤 ÷ VWAP_0905)。
  Yahoo 缺量 miss＝官方日量 − Yahoo 全日量（＝09:00 K 棒〔開盤集合競價＋09:00～09:05 連續〕＋收盤集合競價＋缺根）。
"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RAW, DATA = HERE / 'raw', HERE / 'data'
CAL = (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-06-30'))
VAL = (pd.Timestamp('2025-07-01'), pd.Timestamp('2026-04-01'))
BARS = [f'09:{m:02d}' for m in range(5, 60, 5)]

def period(d):
    d = pd.to_datetime(d)
    return np.where((d >= CAL[0]) & (d <= CAL[1]), 'CAL', np.where((d >= VAL[0]) & (d <= VAL[1]), 'VAL', 'OUT'))

def _feat_l1():
    F = pd.read_parquet(DATA / 'feat_all_2023.parquet')
    F = F.sort_values(['code', 'date'])
    g = F.groupby('code')
    for c in ('V60', 'sig60', 'vrank'): F[c + '_l1'] = g[c].shift(1)
    return F

def exec_flags():
    """(date, code) → pick（任一 AB seed 或 V4 在該成交日持有／新進）、new（新進）、drop（被剔除）；以及成交日集合。"""
    P = pd.concat([pd.read_csv(DATA / 'plans_ab.csv', dtype=str), pd.read_csv(DATA / 'plan_v4.csv', dtype=str)])
    pick, new, drop = set(), set(), set()
    for s, g in P.groupby('seed'):
        prev = set()
        for e, pk in zip(g['exec'], g.picks):
            cur = set(pk.split(',')); d = pd.Timestamp(e)
            pick |= {(d, c) for c in cur}; new |= {(d, c) for c in cur - prev}; drop |= {(d, c) for c in prev - cur}
            prev = cur
    days = sorted({pd.Timestamp(e) for e in P['exec']})
    return pick, new, drop, days

def scores_long():
    S = pd.read_parquet(DATA / 'scores_v4.parquet')
    S['signal'] = pd.to_datetime(S.signal); S['exec'] = pd.to_datetime(S['exec'])
    return S

def stockdays(keep_bars=True):
    H = pd.read_parquet(RAW / 'tw5m_head.parquet'); Dd = pd.read_parquet(RAW / 'tw5m_daily.parquet')
    H['date'] = pd.to_datetime(H.date); Dd['date'] = pd.to_datetime(Dd.date)
    C = H.pivot_table(index=['code', 'date'], columns='hm', values='close')
    Q = H.pivot_table(index=['code', 'date'], columns='hm', values='volume')
    Hh = H.pivot_table(index=['code', 'date'], columns='hm', values='high'); Ll = H.pivot_table(index=['code', 'date'], columns='hm', values='low')
    Oy = H[H.hm == '09:00'].set_index(['code', 'date'])['open']
    hm_all = ['09:00'] + BARS
    full = C.reindex(columns=hm_all).notna().all(axis=1) & Q.reindex(columns=hm_all).notna().all(axis=1)
    X = pd.DataFrame(index=C.index)
    X['full12'] = full
    X['y_open'] = Oy.reindex(C.index)
    X['c0900'] = C['09:00']; X['q0900'] = Q['09:00']
    for b in BARS:
        X['q' + b] = Q[b]
    prevc = C['09:00']
    rb = {}
    for b in BARS:
        rb[b] = np.log(C[b] / prevc); prevc = C[b]
    R = pd.DataFrame(rb)
    w5 = ['09:05', '09:10', '09:15', '09:20', '09:25']
    tp = (Hh[w5] + Ll[w5] + C[w5]) / 3
    X['vol_0905_0925'] = Q[w5].sum(axis=1)
    X['vwap_0905'] = (tp * Q[w5]).sum(axis=1) / Q[w5].sum(axis=1).replace(0, np.nan)
    X['c0925'] = C['09:25']; X['c0955'] = C['09:55']
    X = X.reset_index().merge(Dd[['code', 'date', 'vol_day', 'nbar', 'hm_last', 'close_last']], on=['code', 'date'], how='left')
    O = pd.read_parquet(DATA / 'official_daily.parquet'); O['date'] = pd.to_datetime(O.date)
    X = X.merge(O[['code', 'date', 'open', 'high', 'low', 'close', 'vol']].rename(columns={'open': 'o_off', 'high': 'h_off', 'low': 'l_off', 'close': 'c_off', 'vol': 'vol_off'}), on=['code', 'date'], how='left')
    F = _feat_l1()
    X = X.merge(F[['code', 'date', 'V60_l1', 'sig60_l1', 'vrank_l1', 'prevc', 'fin']], on=['code', 'date'], how='left')
    X['V60sh'] = X.V60_l1 / X.prevc
    X['dec'] = np.ceil(X.vrank_l1 / 30)
    X['gap'] = X.o_off / X.prevc - 1
    X['prem'] = np.log(X.o_off / X.vwap_0905)
    X['miss'] = X.vol_off - X.vol_day
    X['period'] = period(X.date)
    pick, new, drop, days = exec_flags()
    key = list(zip(X.date, X.code))
    X['is_exec_day'] = X.date.isin(set(days))
    X['pick'] = [k in pick for k in key]; X['new'] = [k in new for k in key]; X['drop'] = [k in drop for k in key]
    # 綜合分數：最近一次訊號（≤ 當日前一交易日）的百分位
    S = scores_long()[['signal', 'code', 'pct', 'score']].sort_values('signal')
    X = X.sort_values('date')
    X = pd.merge_asof(X, S.rename(columns={'signal': 'sig_date'}).sort_values('sig_date'), left_on='date', right_on='sig_date', by='code',
                      allow_exact_matches=False, direction='backward')
    X = X.sort_values(['code', 'date']).reset_index(drop=True)
    R = R.reset_index().merge(X[['code', 'date']], on=['code', 'date'], how='right')
    return X, R

def universe(X):
    """校準宇宙：前 300 池、上市普通股、非金融、12 根 K 棒齊全、根數 ≥ 50、官方量與 σ 有值、Yahoo 開盤＝官方開盤（±0.5%）。"""
    m = (X.vrank_l1 <= 300) & X.full12 & (X.nbar >= 50) & (X.vol_off > 0) & np.isfinite(X.sig60_l1) & (X.sig60_l1 > 0) & np.isfinite(X.V60sh) & (X.fin == False)
    m &= (np.abs(X.y_open / X.o_off - 1) <= 0.005)
    return m

def mis():
    rows = []
    for f in sorted(glob.glob(str(RAW / 'screener' / 'vol_curve' / '*.csv'))):
        d = pd.read_csv(f, dtype={'code': str}); day = Path(f).stem
        w = d.pivot_table(index='code', columns='time', values='vol_lots', aggfunc='last')
        need = ['09:01', '09:04', '09:07', '09:10', '09:28', '09:31', '13:21', '13:24']
        if not all(c in w for c in need): continue
        w = w[need].dropna(); w['date'] = pd.Timestamp(day); rows.append(w.reset_index())
    V = pd.concat(rows, ignore_index=True)
    for c in ['09:01', '09:04', '09:07', '09:10', '09:28', '09:31', '13:21', '13:24']: V[c] = V[c] * 1000.0
    V['OA'] = (V['09:01'] - (V['09:04'] - V['09:01']) / 3).clip(lower=0)
    V['V0905'] = V['09:04'] + (V['09:07'] - V['09:04']) / 3
    V['F5'] = (V['V0905'] - V['OA']).clip(lower=0)
    V['bar0905'] = (V['09:10'] - V['V0905']).clip(lower=0)
    h = pd.read_csv(RAW / 'screener' / 'history.csv.gz', dtype={'code': str})
    sl = pd.read_csv(RAW / 'screener' / 'stock_list.csv', dtype=str)
    h = h[h.code.isin(set(sl[sl.market == 'TWSE'].code)) & (h.code.str.len() == 4) & ~h.code.str.startswith('0')].copy()
    h['date'] = pd.to_datetime(h.date); h = h.sort_values(['code', 'date'])
    h['value'] = h.volume * h.close
    g = h.groupby('code')
    h['V60_l1'] = g['value'].transform(lambda s: s.shift(1).rolling(60, min_periods=40).mean())
    h['ret'] = g['close'].pct_change()
    h['sig60_l1'] = g['ret'].transform(lambda s: s.shift(1).rolling(60, min_periods=40).std())
    h['prevc'] = g['close'].shift(1)
    base = (h.V60_l1 >= 1e7) & (h.prevc >= 10)
    h['vrank_l1'] = h.assign(v=np.where(base, h.V60_l1, np.nan)).groupby('date')['v'].rank(ascending=False, method='first')
    h['vrank_all'] = h.groupby('date')['V60_l1'].rank(ascending=False, method='first')     # 不設門檻的名次（池外股的特徵用）
    V = V.merge(h[['code', 'date', 'open', 'close', 'volume', 'prevc', 'V60_l1', 'sig60_l1', 'vrank_l1', 'vrank_all']], on=['code', 'date'], how='left')
    V['CA'] = (V.volume - V['13:24'] - (V['13:24'] - V['13:21']) / 3).clip(lower=0)
    V['V60sh'] = V.V60_l1 / V.prevc
    V['gap'] = V.open / V.prevc - 1
    V['dec'] = np.ceil(V.vrank_l1 / 30)
    V['oa_share'] = V.OA / V.volume; V['ca_share'] = V.CA / V.volume; V['f5_share'] = V.F5 / V.volume
    V['rho'] = V.F5 / V.bar0905.replace(0, np.nan)
    return V

def oddlot():
    o = pd.read_csv(RAW / 'screener' / 'extras' / 'oddlot_hist.csv.gz', dtype={'code': str})
    o = o[(o.code.str.len() == 4) & ~o.code.str.startswith('0') & (o.market == 'TWSE')].copy()
    o['date'] = pd.to_datetime(o.date)
    h = pd.read_csv(RAW / 'screener' / 'history.csv.gz', dtype={'code': str}); h['date'] = pd.to_datetime(h.date)
    h = h.sort_values(['code', 'date']); h['value'] = h.volume * h.close
    g = h.groupby('code'); h['V60_l1'] = g['value'].transform(lambda s: s.shift(1).rolling(60, min_periods=40).mean()); h['prevc'] = g['close'].shift(1)
    base = (h.V60_l1 >= 1e7) & (h.prevc >= 10)
    h['vrank_l1'] = h.assign(v=np.where(base, h.V60_l1, np.nan)).groupby('date')['v'].rank(ascending=False, method='first')
    m = o.merge(h[['code', 'date', 'open', 'close', 'volume', 'V60_l1', 'vrank_l1', 'prevc']], on=['code', 'date'], how='inner')
    m['buy_cost'] = m.ask / m.close - 1; m['sell_cost'] = 1 - m.bid / m.close
    m['dev_close'] = m.price / m.close - 1; m['dev_open'] = m.price / m.open - 1
    m['odd_share_of_day'] = m.amount / m.value if 'value' in m else m.amount / (m.volume * m.close)
    return m
