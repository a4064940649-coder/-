#!/usr/bin/env python3
"""反方審查 o15（成本與可成交）：只 import erlib，只讀 ERRATUM2.json，不改研究員檔案，不寫入任何檔案。
檢查 U_opt 放回槽位的可成交性：進場日漲停、持有期內停止交易／下市（換股或現金收購結算）、併購待結（低波動）、
容量（V60 對 30 萬～1,000 萬）、周轉變化、處置資料覆蓋、營收公告期限與訊號日。用法：python3 -I -B review_exec_o15.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import erlib as E
import numpy as np, pandas as pd
K, V = E.K, E.V
ER = json.loads((HERE / 'ERRATUM2.json').read_text(encoding='utf-8'))
ins = ER['U_opt_inserted']
S = E.VU('opt'); pl = S.plan(10, 'official')
B = E.VU('base'); plb = B.plan(10, 'official')
IDX = K.IDX
sig = [d for (_, _, _, d) in pl]
ent = {str(d.date()): e for (e, _, _, d) in pl}
nxt = {}
for i, (e, _, _, d) in enumerate(pl):
    nxt[str(d.date())] = pl[i + 1][0] if i + 1 < len(pl) else len(IDX) - 1
out = {}
# 1. 處置資料覆蓋
out['disposal_first_start'] = str(K.DSP['start'].min().date())
# 2. 逐槽位可成交性
rows = []
for r in ins:
    d = r['sig']; c = r['code']; j = E.CI[c]; e = ent[d]; x = nxt[d]
    t = e - 1
    gu = bool(K.gap_up_open(e)[j])
    tr = K.TRADED[e:x + 1, j]
    stop_in_hold = bool(not tr.all())
    lt = E.LT[c]
    terminal = bool(lt < IDX[x])          # 下一次換股進場前已最後交易（持有中下市）
    w = K.TRADED[max(0, t - 59):t + 1, j]
    rr = K.Rn[max(0, t - 59):t + 1, j]
    sd = float(np.nanstd(rr[w])) if w.sum() > 20 else np.nan
    v60 = float(S.V60[t, j])
    days_to_last = int(((IDX > IDX[t]) & (IDX <= lt)).sum())
    rows.append(dict(sig=d, code=c, name=r['name'], cause=r['cause'], gap_up_entry=gu, stop_in_hold=stop_in_hold,
                     terminal=terminal, sd60=sd, v60=v60, tdays_to_last_trade=days_to_last))
R = pd.DataFrame(rows)
R['y'] = R.sig.str[:4]
D12 = R[R.sig < '2013-01-01']
out['n_inserted'] = len(R); out['n_2006_2012'] = len(D12)
out['gap_up_entry'] = int(R.gap_up_entry.sum())
out['stop_in_hold'] = int(R.stop_in_hold.sum()); out['terminal_in_hold'] = int(R.terminal.sum())
out['terminal_rows'] = R[R.terminal][['sig', 'code', 'name', 'cause', 'tdays_to_last_trade']].to_dict(orient='records')
out['within_120td_of_last_trade'] = int((R.tdays_to_last_trade <= 120).sum())
out['within_120td_2006_2012'] = int((D12.tdays_to_last_trade <= 120).sum())
out['lowvol_sd60_lt_1pct'] = int((R.sd60 < 0.01).sum())
out['lowvol_rows'] = R[R.sd60 < 0.01][['sig', 'code', 'name', 'cause', 'sd60', 'tdays_to_last_trade']].round(4).to_dict(orient='records')
# 若把「持有期內下市」與「距最後交易 ≤ 60 交易日且低波動（併購待結）」視為散戶實際不會／不能有效持有，槽位剩多少
bad = R.terminal | ((R.tdays_to_last_trade <= 60) & (R.sd60 < 0.01))
out['exclude_terminal_or_dealpending'] = dict(u_2006_2012=int((~bad & (R.sig < '2013-01-01')).sum()),
                                              share_2006_2012=float((~bad & (R.sig < '2013-01-01')).sum() / 810),
                                              u_2006_2013=int((~bad).sum()), share_2006_2013=float((~bad).sum() / 930))
# 容量：V60（60 日均成交值）與 1,000 萬帳戶每檔 100 萬部位
out['v60_min_inserted'] = float(R.v60.min()); out['v60_p10_inserted'] = float(R.v60.quantile(0.1))
out['pos_1000w_over_v60_max'] = float((1e6 / R.v60).max())
# 3. 周轉：每期名單更替數
def turnover(P):
    ch = []
    prev = None
    for e, pk, el, d in P:
        if not (E.LO <= d <= E.HI): prev = set(pk); continue
        s = set(pk)
        if prev is not None: ch.append(len(s - prev))
        prev = s
    return ch
tb, to = turnover(plb), turnover(pl)
out['turnover_names_per_rebalance'] = dict(base=float(np.mean(tb)), U_opt=float(np.mean(to)),
                                          base_sum=int(sum(tb)), U_opt_sum=int(sum(to)))
# 4. 營收公告期限 vs 訊號日（2006～2013）
dc = pd.Series([V.deadline_case(d) for d in sig if E.LO <= d <= E.HI]).value_counts().to_dict()
out['deadline_case_2006_2013'] = dc
# 5. 5854 是否銀行（不在 FINMASK）且其入選期間
out['5854_in_FINMASK'] = bool(V.FINMASK[E.CI['5854']]) if '5854' in E.CI else None
out['5854_lastrev'] = str(E.LASTREV.get('5854', '')); out['5854_firstrev'] = str(E.FIRSTREV.get('5854', ''))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(R.groupby('cause')[['terminal', 'stop_in_hold']].sum())
print(R.sort_values('tdays_to_last_trade').head(40).to_string())
