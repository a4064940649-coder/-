#!/usr/bin/env python3
"""可執行審查（只描述、不判定）：下市金融股被選中的槽位在當時是否可成交——進場日開盤漲停、持有期內停牌、距最後交易日、
60 日報酬標準差（併購待結的低波動）、V60 與 100 萬／300 萬／1,000 萬帳戶每檔部位占比；最後交易日後的結算方式（換股／合併）。
只讀 FM_COUNTS.json，結果寫 review_exec_fm.json。用法：python3 -I -B review_exec_fm.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freg as R
R.check_reg('ERRATUM_SPEC_FM')
import flib as L
import numpy as np, pandas as pd
K = L.K
FM = json.loads((HERE / 'FM_COUNTS.json').read_text(encoding='utf-8'))
S52, V60 = L.V.rolling_signals('gapfix')
SETTLE = {'5854': '股份轉換為 5880 合庫金（研究員記憶，未查證）', '6004': '與元大證合併（研究員記憶，未查證）', '6012': '股份轉換為開發金子公司（研究員記憶，未查證）'}
rows = []
seen = set()
for lst in ('C0', 'U_opt'):
    for d, c, rk in FM['direct']['F2'][lst]:
        if (lst, d, c) in seen: continue
        seen.add((lst, d, c))
        j = L.CI[c]; t = K.IDX.get_loc(pd.Timestamp(d)); e = t + 1
        w = np.where(K.TRADED[:, j])[0]; lt = w[-1]
        r = K.Rn[max(0, t - 59):t + 1, j]; m = K.TRADED[max(0, t - 59):t + 1, j]
        sd = float(np.nanstd(r[m])) if m.sum() > 20 else None
        v = float(V60[t, j])
        rows.append(dict(list=lst, sig=d, code=c, rank=rk, entry=str(K.IDX[e].date()), open_entry=float(K.On[e, j]), gap_up_entry=bool(K.gap_up_open(e)[j]),
                         last_trade=str(K.IDX[lt].date()), tdays_to_last_trade=int(lt - t), sd60=sd, v60_mn=round(v / 1e6, 1),
                         pos_share_of_v60={'100萬': round(1e5 / v, 5), '300萬': round(3e5 / v, 5), '1000萬': round(1e6 / v, 5)},
                         deal_pending_proxy=bool(sd is not None and sd < 0.01 and (lt - t) <= 60), settle=SETTLE.get(c, '')))
D = pd.DataFrame(rows)
out = dict(rows=rows, n=len(rows), n_gap_up=int(D.gap_up_entry.sum()), n_within_60td_of_last=int((D.tdays_to_last_trade <= 60).sum()),
           n_deal_pending_proxy=int(D.deal_pending_proxy.sum()), max_pos_share_1000w=float(max(r['pos_share_of_v60']['1000萬'] for r in rows)),
           note='每檔部位＝帳戶 ÷ 10；占 V60 比例都很小。直接污染槽位在當時都可成交，所以「規則違反」是真的會發生在實盤上的違反，不是無法成交的紙上槽位。')
(HERE / 'review_exec_fm.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(D.to_string()); print({k: v for k, v in out.items() if k != 'rows'})
