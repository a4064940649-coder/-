#!/usr/bin/env python3
"""校準資料（2024-02～2026-10）的共用前處理：只看執行量，不看任何策略報酬。
  load_5m()：Yahoo 5 分 K（曾入選股），每股每日一列：bar0905 量（09:05–09:10 連續交易量）、p910（09:05 K 收盤＝9:10 價）、開盤、Yahoo 全日量、
             60 日 Yahoo 均量（GOOD 日，≥40 筆）、前收、跳空、池內成交值名次、σ60、是否為某 seed 的成交日。
  load_vc()：vol_curve（2026-10 分鐘累積量）＋官方日 K（history.csv.gz）：OA_central＝V(09:01)−(V(09:04)−V(09:01))/3、OA_upper＝V(09:01)、
             bar0905_equiv＝V(09:10)−[V(09:04)+(V(09:07)−V(09:04))/3]；官方 60 日均成交值與上市股名次。
  load_odd()：oddlot_hist（2026-07-13 起）× 官方日 K：買成本＝ask/close−1、賣成本＝1−bid/close、成交價偏離＝price/close−1。
"""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RAW, DATA = HERE / 'raw', HERE / 'data'
YAHOO_TO_OFFICIAL = None   # 由 load_5m() 估計（2025-09-24 起與官方量重疊期的中位數）

def _hist():
    h = pd.read_csv(RAW / 'screener' / 'history.csv.gz', dtype={'code': str})
    sl = pd.read_csv(RAW / 'screener' / 'stock_list.csv', dtype=str)
    h = h[h.code.isin(set(sl[sl.market == 'TWSE'].code)) & (h.code.str.len() == 4) & ~h.code.str.startswith('0')]
    h['date'] = pd.to_datetime(h['date'])
    h['value'] = h.volume * h.close
    h = h.sort_values(['code', 'date'])
    h['V60_off'] = h.groupby('code')['value'].transform(lambda s: s.shift(1).rolling(60, min_periods=40).mean())
    h['prevc'] = h.groupby('code')['close'].shift(1)
    base = (h.V60_off >= 1e7) & (h.close.groupby(h.code).shift(1) >= 10)
    h['vrank_off'] = np.where(base, h.assign(v=np.where(base, h.V60_off, np.nan)).groupby('date')['v'].rank(ascending=False, method='first'), np.nan)
    return h

def load_5m():
    H = pd.read_parquet(RAW / 'tw_stocker_head.parquet'); Dd = pd.read_parquet(RAW / 'tw_stocker_daily.parquet')
    w = H.pivot_table(index=['code', 'date'], columns='hm', values=['volume', 'close', 'open'])
    X = pd.DataFrame({'bar0905': w[('volume', '09:05')], 'bar0910': w[('volume', '09:10')], 'p910': w[('close', '09:05')],
                      'p905': w[('close', '09:00')], 'y_open': w[('open', '09:00')]}).reset_index()
    X = X.merge(Dd[['code', 'date', 'vol_day', 'nbar']], on=['code', 'date'], how='left')
    X['date'] = pd.to_datetime(X['date'])
    X = X.sort_values(['code', 'date'])
    X['ADV60_y'] = X.groupby('code')['vol_day'].transform(lambda s: s.shift(1).rolling(60, min_periods=40).mean())
    # 面板欄位（原始價、σ60、名次）
    O = pd.read_parquet(DATA / 'px_open.parquet'); PC = pd.read_parquet(DATA / 'prevclose.parquet')
    F = pd.read_parquet(DATA / 'feat_daily.parquet')
    F = F.sort_values(['code', 'date']); F['vrank_l1'] = F.groupby('code')['vrank'].shift(1); F['sig_l1'] = F.groupby('code')['sig60'].shift(1)
    F['V60_l1'] = F.groupby('code')['V60'].shift(1)
    X = X.merge(F[['date', 'code', 'vrank_l1', 'sig_l1', 'V60_l1']], on=['date', 'code'], how='left')
    os_ = O.stack().rename('open_p').reset_index().rename(columns={'level_1': 'code'}); os_.columns = ['date', 'code', 'open_p']
    ps_ = PC.stack().rename('prevc').reset_index(); ps_.columns = ['date', 'code', 'prevc']
    X = X.merge(os_, on=['date', 'code'], how='left').merge(ps_, on=['date', 'code'], how='left')
    X['gap'] = X.open_p / X.prevc - 1
    X['drift910'] = X.p910 / X.y_open - 1
    # 成交日旗標（任一 seed 或 V4 的 exec 日、且為入選股）
    P = pd.concat([pd.read_csv(DATA / 'plans_ab.csv', dtype=str), pd.read_csv(DATA / 'plan_v4.csv', dtype=str)])
    ex = {(pd.Timestamp(e), c) for e, pk in zip(P['exec'], P.picks) for c in pk.split(',')}
    X['is_exec'] = [(d, c) in ex for d, c in zip(X.date, X.code)]
    h = _hist()
    m = X.merge(h[['code', 'date', 'volume']], on=['code', 'date'], how='inner')
    global YAHOO_TO_OFFICIAL
    YAHOO_TO_OFFICIAL = float((m.vol_day / m.volume).replace([np.inf, -np.inf], np.nan).dropna().median())
    X['r_bar'] = X.bar0905 / X.ADV60_y * YAHOO_TO_OFFICIAL      # 09:05–09:10 量 ÷ 官方口徑 ADV60
    return X

def load_vc():
    import glob
    rows = []
    for f in sorted(glob.glob(str(RAW / 'screener' / 'vol_curve' / '*.csv'))):
        d = pd.read_csv(f, dtype={'code': str}); day = Path(f).stem
        w = d.pivot_table(index='code', columns='time', values='vol_lots', aggfunc='last')
        need = ['09:01', '09:04', '09:07', '09:10']
        if not all(c in w for c in need): continue
        w = w[need].dropna()
        w['date'] = pd.Timestamp(day); rows.append(w.reset_index())
    V = pd.concat(rows, ignore_index=True)
    V['OA_central'] = (V['09:01'] - (V['09:04'] - V['09:01']) / 3).clip(lower=0) * 1000
    V['OA_upper'] = V['09:01'] * 1000
    V['bar0905_eq'] = (V['09:10'] - (V['09:04'] + (V['09:07'] - V['09:04']) / 3)).clip(lower=0) * 1000
    h = _hist()
    V = V.merge(h[['code', 'date', 'volume', 'close', 'open', 'prevc', 'V60_off', 'vrank_off', 'value']], on=['code', 'date'], how='left')
    V['oa_share_day'] = V.OA_central / V.volume
    V['ratio_central'] = V.OA_central * V.open / V.V60_off
    V['ratio_upper'] = V.OA_upper * V.open / V.V60_off
    V['gap'] = V.open / V.prevc - 1
    return V

def load_odd():
    o = pd.read_csv(RAW / 'screener' / 'oddlot_hist.csv.gz', dtype={'code': str})
    o = o[(o.code.str.len() == 4) & ~o.code.str.startswith('0')]
    o['date'] = pd.to_datetime(o['date'])
    h = _hist()
    m = o.merge(h[['code', 'date', 'close', 'open', 'V60_off', 'vrank_off']], on=['code', 'date'], how='inner')
    m['buy_cost'] = m.ask / m.close - 1
    m['sell_cost'] = 1 - m.bid / m.close
    m['dev_close'] = m.price / m.close - 1
    m['dev_open'] = m.price / m.open - 1
    m['hs'] = (m.ask - m.bid) / ((m.ask + m.bid) / 2) / 2
    return m
