#!/usr/bin/env python3
"""第 0 步 (a)-3：2005-01～2021-09-23 處置股歷史名單 disposal_hist.csv（不看任何策略報酬）。
python3 -I -B build_disposal.py

資料搜尋結果：GitHub 上沒有找到 2021-09 以前 TWSE「公布處置有價證券」的封存（見 FINDINGS 資料節）；
新聞鏡像（Intern-in-CTBC 2013～2020）每年只有 23～105 篇提到「處置」，覆蓋率遠低於 90%，不能當名單。
所以用 BACKLOG 的後備：以價格規則近似證交所「公布注意交易資訊→處置」的機制，在 2021-09-24～2023-12 用真實名單
（disposal_5y，只取面板內＝上市代碼）調整版本（挑選），2024-01～2026-10 只報告（驗證），版本與結果全部寫進 iterations.json。

近似規則的元件（以含息還原收盤 MYADJ 與加權指數 TAIEX 計算；t 日收盤後判定，處置從 t+1 起 10 個交易日）：
  c1：6 日累積漲幅 > A6，且比加權指數 6 日漲幅多 ≥ 20pp（注意交易資訊第一款的近似）
  c2：30 日漲幅 > 100% 且超額 ≥ 85pp；60 日 > 130% 且超額 ≥ 110pp；90 日 > 160% 且超額 ≥ 135pp（第二款近似）
  c3：6 日累積成交值 ≥ 60 日平均的 5 倍，且 6 日漲幅 > 25%（量增價漲近似）
  觸發（處置）：連續 3 日 c1、或連續 5 日任一款、或最近 10 日有 6 日任一款、或最近 30 日有 12 日任一款（只用 c1～c3 近似「第一～八款」）
評估單位：流動性合格股票日（60 日均成交值 ≥ 1,000 萬且收盤 ≥ 10，即主策略前 300 池的必要條件）上「是否處置中」。
選版規則（事前定）：挑選期 F1 最高者；F1 差 < 0.01 時取召回率較高者。
"""
import json, hashlib, datetime, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import o3_core as O
K = O.K
O.apply_patch(require_freeze=False)

IDX = K.IDX; N = K.N
A = K.MYADJ.ffill()
TX = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close']
TX = TX.reindex(IDX.union(TX.index)).ffill().reindex(IDX)
Vv = K.V
v60 = Vv.rolling(60, min_periods=50).mean()
LIQ = (v60 >= 1e7) & (K.C >= 10)
TRD = pd.DataFrame(K.TRADED, index=IDX, columns=K.COLS)

def chg(n):
    return A / A.shift(n) - 1, TX / TX.shift(n) - 1
R6, M6 = chg(6); R30, M30 = chg(30); R60, M60 = chg(60); R90, M90 = chg(90)
EX6 = R6.sub(M6, axis=0)
V6 = Vv.rolling(6, min_periods=6).sum(); V60s = Vv.rolling(60, min_periods=50).mean() * 6

def comp(a6=0.32, use_c2=True, use_c3=True, ex6=0.20):
    c1 = (R6 > a6) & (EX6 >= ex6)
    c2 = ((R30 > 1.0) & (R30.sub(M30, axis=0) >= 0.85)) | ((R60 > 1.3) & (R60.sub(M60, axis=0) >= 1.10)) | ((R90 > 1.6) & (R90.sub(M90, axis=0) >= 1.35))
    c3 = (V6 >= 5 * V60s) & (R6 > 0.25)
    anyc = c1 | (c2 if use_c2 else False) | (c3 if use_c3 else False)
    return (c1 & TRD), (anyc & TRD)

def trigger(c1, anyc, mode):
    c1f, af = c1.astype(float), anyc.astype(float)
    t1 = c1f.rolling(3, min_periods=3).sum() >= 3
    if mode == 'c1_3consec':
        return t1
    if mode == 'c1_3_any5':
        return t1 | (af.rolling(5, min_periods=5).sum() >= 5)
    t5 = af.rolling(5, min_periods=5).sum() >= 5
    t610 = af.rolling(10, min_periods=10).sum() >= 6
    t1230 = af.rolling(30, min_periods=30).sum() >= 12
    return t1 | t5 | t610 | t1230

def events_from(trig, length=10):
    """觸發日 t → 處置期 t+1～t+length（交易日）；處置中再觸發不另起（期滿後才可再觸發）。"""
    T = trig.to_numpy(); ev = []
    for j in range(N):
        tt = np.where(T[:, j])[0]
        busy = -1
        for t in tt:
            if t <= busy: continue
            s, e = t + 1, min(t + length, len(IDX) - 1)
            if s >= len(IDX): continue
            ev.append((K.COLS[j], IDX[t], IDX[s], IDX[e])); busy = e
    return pd.DataFrame(ev, columns=['code', 'trigger', 'start', 'end'])

def mask_from_events(E):
    M = np.zeros((len(IDX), N), bool)
    for c, s, e in zip(E['code'], E['start'], E['end']):
        j = K.CI.get(c)
        if j is None: continue
        M[IDX.get_loc(s):IDX.get_loc(e) + 1, j] = True
    return pd.DataFrame(M, index=IDX, columns=K.COLS)

# 真實名單（面板代碼＝上市），日期對齊到面板交易日
R = O.DSP5.copy()
R = R[R['code'].isin(K.CI)]
R['start'] = [IDX[IDX >= s][0] if (IDX >= s).any() else pd.NaT for s in R['start']]
R['end'] = [IDX[IDX <= e][-1] for e in R['end']]
R = R.dropna(subset=['start'])
MR = mask_from_events(R)

def score(MP, a, b):
    sel = (IDX >= pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
    L = LIQ.to_numpy()[sel] & TRD.to_numpy()[sel]
    p, r = MP.to_numpy()[sel] & L, MR.to_numpy()[sel] & L
    tp = int((p & r).sum()); fp = int((p & ~r).sum()); fn = int((~p & r).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan; rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else np.nan
    # 事件層：真實事件（流動性合格）中，起日 ±3 交易日內有近似事件起日者
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': prec, 'recall': rec, 'F1': f1}

def event_recall(EP, a, b):
    Rr = R[(R['start'] >= pd.Timestamp(a)) & (R['start'] <= pd.Timestamp(b))]
    ok = 0; n = 0
    for c, s in zip(Rr['code'], Rr['start']):
        j = K.CI[c]; t = IDX.get_loc(s)
        if not (LIQ.iat[t - 1, j] if t > 0 else False): continue
        n += 1
        x = EP[(EP['code'] == c)]
        if ((x['start'] >= IDX[max(t - 3, 0)]) & (x['start'] <= IDX[min(t + 3, len(IDX) - 1)])).any(): ok += 1
    return ok / n if n else np.nan, n

SEL = ('2021-09-24', '2023-12-31'); VAL = ('2024-01-01', '2026-10-22')
VERS = {
    'Q1_bear_c1_3consec': dict(a6=0.32, use_c2=False, use_c3=False, mode='c1_3consec'),
    'Q2_c1c2_alltriggers': dict(a6=0.32, use_c2=True, use_c3=False, mode='all'),
    'Q3_c1c2c3_alltriggers': dict(a6=0.32, use_c2=True, use_c3=True, mode='all'),
    'Q4_Q3_a6_25': dict(a6=0.25, use_c2=True, use_c3=True, mode='all'),
    'Q5_Q3_a6_28_ex15': dict(a6=0.28, use_c2=True, use_c3=True, mode='all', ex6=0.15),
    'Q6_c1_3consec_or_c1c2_5consec': dict(a6=0.32, use_c2=True, use_c3=False, mode='c1_3_any5'),
    'Q7_c1_3consec_a6_30_ex18': dict(a6=0.30, use_c2=False, use_c3=False, mode='c1_3consec', ex6=0.18),
}
its = []; built = {}
for k, v in VERS.items():
    c1, anyc = comp(v['a6'], v['use_c2'], v['use_c3'], v.get('ex6', 0.20))
    E = events_from(trigger(c1, anyc, v['mode']))
    MP = mask_from_events(E); built[k] = (E, MP)
    s1, s2 = score(MP, *SEL), score(MP, *VAL)
    er1, n1 = event_recall(E, *SEL); er2, n2 = event_recall(E, *VAL)
    pre = E[E['start'] < O.D5_START]
    yr = pre.groupby(pre['start'].dt.year).size().to_dict()
    its.append({'version': k, 'spec': v, 'select_2021-09-24_2023': s1, 'event_recall_select': er1, 'n_real_events_select': n1,
                'validate_2024_2026': s2, 'event_recall_validate': er2, 'n_real_events_validate': n2,
                'n_events_2005_2021-09': int(len(pre)), 'events_per_year_pre': {int(a): int(b) for a, b in yr.items()}})
    print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in s1.items()}, 'evrec', round(er1, 3), '| val', {x: (round(y, 3) if isinstance(y, float) else y) for x, y in s2.items()}, round(er2, 3), flush=True)
# 選版
best = sorted(its, key=lambda x: (-(round(x['select_2021-09-24_2023']['F1'], 2)), -x['select_2021-09-24_2023']['recall']))[0]
bk = best['version']
E, MP = built[bk]
real_y = R.groupby(R['start'].dt.year).size().to_dict()
H = E[E['start'] < O.D5_START].copy(); H['rule'] = bk
H['start'] = H['start'].dt.date.astype(str); H['end'] = H['end'].dt.date.astype(str); H['trigger'] = H['trigger'].dt.date.astype(str)
H.to_csv(HERE / 'disposal_hist.csv', index=False)
# 召回率推估的漏網名額（all_plus 用）：每個主策略訊號日，流動性合格且處置中的「真實」檔數 ÷ 近似檔數 的比例，在挑選期估計
sel = (IDX >= pd.Timestamp(SEL[0])) & (IDX <= pd.Timestamp(SEL[1]))
L = (LIQ.to_numpy() & TRD.to_numpy())[sel]
n_real = (MR.to_numpy()[sel] & L).sum(axis=1).mean(); n_prox = (MP.to_numpy()[sel] & L).sum(axis=1).mean()
tp_day = (MR.to_numpy()[sel] & MP.to_numpy()[sel] & L).sum(axis=1).mean()
meta = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'chosen': bk, 'iterations': its,
        'real_events_per_year_panel_codes': {int(a): int(b) for a, b in real_y.items()},
        'per_day_liquid_real_mean_select': float(n_real), 'per_day_liquid_proxy_mean_select': float(n_prox), 'per_day_tp_mean_select': float(tp_day),
        'missed_per_day_mean_select': float(n_real - tp_day),
        'sha256': hashlib.sha256((HERE / 'disposal_hist.csv').read_bytes()).hexdigest()}
(HERE / 'disposal_hist_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print('chosen', bk, 'n', len(H), 'real/day', round(n_real, 2), 'proxy/day', round(n_prox, 2), 'tp/day', round(tp_day, 2))
print('real per year', meta['real_events_per_year_panel_codes'])
print('proxy per year', best['events_per_year_pre'])
