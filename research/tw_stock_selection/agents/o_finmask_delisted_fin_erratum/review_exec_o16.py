#!/usr/bin/env python3
"""反方可執行審查 o16（成本與可成交角度）。只讀研究員輸出與面板，不改任何研究員檔案，不寫 results.json。
檢查 FM_COUNTS 六組異動的「換出／換入」兩側（不只直接污染 14 槽）：
 1. 進場日（訊號 t+1）開盤漲停、收盤仍在漲停（鎖死代理，無 low 欄）、進場日未交易
 2. 出場日（下一訊號 t+1）開盤跌停（賣不掉代理）、持有期內停牌日數
 3. 研究中心官方全歷史處置事件期（o_disposal_official_fullhist）：訊號日、進場日、持有期內是否在處置期（需預收款券／分盤）
 4. 容量：60 日均成交值 V60，1,000 萬帳戶每檔 100 萬占比
 5. 描述性逐槽毛報酬（開盤到下一進場開盤，未扣成本；只為檢查「計數比例」是否嚴重低估影響，非判定）
 6. 換檔額外成本：每槽一次來回 0.3% 稅＋手續費 2×0.0855%＋滑價 2×0.1%～0.3%
用法：python3 -I -B review_exec_o16.py（輸出寫到 scratchpad，不寫進研究員資料夾）"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'independent_audit'))
import ia_core as K
sys.path.pop(0)
import numpy as np, pandas as pd

FM = json.loads((HERE / 'FM_COUNTS.json').read_text(encoding='utf-8'))
C0L = FM['lists']['C0|C0']
SIG = sorted(C0L)
IDX = K.IDX
def tix(d): return IDX.get_loc(pd.Timestamp(d))
V60 = pd.DataFrame(K.V.to_numpy(), index=IDX).rolling(60, min_periods=20).mean().to_numpy()
ADJF = (K.ADJ / K.C).to_numpy()
AOPEN = K.On * ADJF
DSP = pd.read_csv(AG / 'o_disposal_official_fullhist' / 'universe_flags.csv', dtype={'code': str}, parse_dates=['start', 'end'])

def limit_down_open(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    ref = K.ref_price(t)
    raw = ref * (1 - lim); tk = K.tick(raw)
    ld = np.ceil(raw / tk - 1e-9) * tk
    return np.isfinite(K.On[t]) & np.isfinite(ref) & (K.On[t] <= ld + 1e-6)

def lock_up_close(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    lu = K.limit_up_price(K.ref_price(t), lim)
    return np.isfinite(K.Cn[t]) & (K.Cn[t] >= lu - 1e-6) & (K.On[t] >= lu - 1e-6)

def in_disp(c, a, b):
    s = DSP[DSP.code == c]
    return bool(((s.start <= pd.Timestamp(b)) & (s.end >= pd.Timestamp(a))).any())

def slot(c, d, side, blk):
    j = K.CI[c]; t = tix(d); e = t + 1
    k = SIG.index(d)
    e2 = tix(SIG[k + 1]) + 1 if k + 1 < len(SIG) else min(e + 21, len(IDX) - 1)
    hold = K.TRADED[e:e2, j]
    gu = bool(K.gap_up_open(e)[j]); lk = bool(lock_up_close(e)[j])
    ld = bool(limit_down_open(e2)[j]) if K.TRADED[e2, j] else None
    r = float(AOPEN[e2, j] / AOPEN[e, j] - 1) if np.isfinite(AOPEN[e2, j]) and np.isfinite(AOPEN[e, j]) else None
    if r is None and np.isfinite(AOPEN[e, j]):
        w = np.where(K.TRADED[e:e2, j])[0]; lt = e + w[-1]
        r = float(K.ADJ.iat[lt, j] / AOPEN[e, j] - 1)
    v = float(V60[t, j])
    return dict(block=blk, side=side, sig=d, code=c, entry=str(IDX[e].date()), exit=str(IDX[e2].date()),
                traded_entry=bool(K.TRADED[e, j]), gap_up_open=gu, lock_up_proxy=lk, exit_limit_down_open=ld,
                halt_days_in_hold=int((~hold).sum()), disp_sig=in_disp(c, d, d), disp_entry=in_disp(c, IDX[e], IDX[e]),
                disp_any_hold=in_disp(c, IDX[e], IDX[e2]), v60_mn=round(v / 1e6, 1), share_100w=round(1e6 / v, 4),
                gross_ret=None if r is None else round(r, 4))

rows = []
for blk, B in FM['blocks'].items():
    for rw in B['rows']:
        for c in rw['out']: rows.append(slot(c, rw['sig'], 'out', blk))
        for c in rw['in_']: rows.append(slot(c, rw['sig'], 'in', blk))
D = pd.DataFrame(rows)
summ = {}
for blk in FM['blocks']:
    x = D[D.block == blk]
    o, i = x[x.side == 'out'], x[x.side == 'in']
    summ[blk] = dict(n_slots=int(len(o)), gap_up_in=int(i.gap_up_open.sum()), lock_in=int(i.lock_up_proxy.sum()),
                     untraded_entry_in=int((~i.traded_entry).sum()), exit_ld_any=int(x.exit_limit_down_open.fillna(False).astype(bool).sum()),
                     halt_slots=int((x.halt_days_in_hold > 0).sum()), disp_any_hold=int(x.disp_any_hold.sum()),
                     disp_entry=int(x.disp_entry.sum()), max_share_100w=float(x.share_100w.max()), min_v60_mn=float(x.v60_mn.min()),
                     sum_gross_in_minus_out=round(float(i.gross_ret.sum() - o.gross_ret.sum()), 4),
                     mean_gross_in_minus_out=round(float((i.gross_ret.sum() - o.gross_ret.sum()) / max(len(o), 1)), 4))
# 換檔成本：每異動槽多一次來回（若原名單該檔本來就續抱、新名單換掉，最多多一次來回；上界）
rt_cost = {s: round(0.003 + 2 * 0.001425 * 0.6 + 2 * s, 5) for s in (0.001, 0.002, 0.003)}
# 全期 930 槽 × 每槽 1/10 權重：異動槽造成的累積毛報酬差（等權 10 檔，每槽權重 0.1）
for blk, s in summ.items():
    s['portfolio_cum_gross_diff_pct'] = round(100 * 0.1 * s['sum_gross_in_minus_out'], 3)
    s['extra_cost_upper_pct_slip0.3'] = round(100 * 0.1 * s['n_slots'] * rt_cost[0.003], 3)
out = dict(rows=rows, summary=summ, roundtrip_cost=rt_cost,
           note='描述性；gross_ret 為進場開盤到下一進場開盤（還原），未扣成本。exit_limit_down_open 只看出場日開盤。lock_up_proxy＝開盤與收盤都在漲停價（無 low 欄，近似）。')
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
(SCR / 'review_exec_o16_out.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.set_option('display.width', 250)
print(D.drop(columns=['exit']).to_string())
print(json.dumps(summ, ensure_ascii=False, indent=1)); print(rt_cost)
