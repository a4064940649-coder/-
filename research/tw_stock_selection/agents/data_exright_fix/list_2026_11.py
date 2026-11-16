#!/usr/bin/env python3
"""第 3 步（名單）：2026-11 名單在修正後漲跌價差下變不變。
直接沿用稽核資料夾的獨立重算程式 audit_live_2026-11/audit_list.py 的第 1～2 節與 build()（讀原始碼、執行到「3. 對名單檔」之前，不改該檔），
只在它建好視窗內 CHG 矩陣之後插入本題的修正。比較：orig（原樣）、main（TWT49U＋前一日無成交改收盤報酬）、
main_plus（main，且 2026-10-30 之後 TWT49U 鏡像沒涵蓋的上市 0 格改收盤報酬）、close（全部 0 格改收盤報酬）。
輸出 scratchpad/exr/list_2026_11.json（compare.py 合併）。python3 -I -B list_2026_11.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from exr_common import TMP, AG, load_twt49u

SRC = (AG / 'audit_live_2026-11' / 'audit_list.py').read_text(encoding='utf-8')
cut = SRC.index('# ---------------------------------------------------------------- 3. 對名單檔')
HOOK = "MKT = np.zeros((T, N), np.int8); MKT[L['t'].to_numpy(), L['stock_id'].map(ci).to_numpy()] = np.where(L['market'].to_numpy() == 'twse', 1, 2)\n"
assert SRC.count(HOOK) == 1
body = SRC[:cut].replace(HOOK, HOOK + "CHG = __FIX__(CHG, C, MKT, use, codes)\n")
body = body.replace("HERE = Path(__file__).resolve().parent", "HERE = Path(%r)" % str(AG / 'audit_live_2026-11'))
EV = load_twt49u()
TWT_LAST = EV['date'].max()


def make_fix(mode, log):
    def fix(CHG, C, MKT, use, codes):
        CHG = CHG.copy()
        T, N = C.shape
        has = np.isfinite(C) & (C > 0)
        CF = pd.DataFrame(np.where(has, C, np.nan)).ffill().to_numpy()
        prevC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
        zero = np.nan_to_num(CHG) == 0
        z = has & np.isfinite(prevC) & zero & (np.abs(C - prevC) > 1e-9)
        prev_tr = np.vstack([np.zeros((1, N), bool), has[:-1]])
        dts = pd.DatetimeIndex(use)
        REFX = np.full((T, N), np.nan)
        di = {d: i for i, d in enumerate(dts)}; ci = {c: j for j, c in enumerate(codes)}
        e = EV[EV['date'].isin(di) & EV['code'].isin(ci)]
        REFX[e['date'].map(di).to_numpy(), e['code'].map(ci).to_numpy()] = e['ref_price'].to_numpy()
        inx = np.isfinite(REFX) & (MKT == 1)
        after = (dts > TWT_LAST)[:, None] & np.ones((1, N), bool)
        log['zero_cells_twse_after_twt_last'] = [(str(dts[t].date()), str(codes[j])) for t, j in zip(*np.nonzero(z & (MKT == 1) & after))]
        if mode == 'orig':
            return CHG
        if mode == 'close':
            CHG[z] = C[z] - prevC[z]; return CHG
        m = has & inx & zero
        with np.errstate(invalid='ignore', divide='ignore'):
            rr = C / REFX
        oob = m & ~((rr >= 0.899) & (rr <= 1.101)); REFX[oob] = prevC[oob]
        CHG[m] = C[m] - REFX[m]
        m3 = z & ~inx & ~prev_tr & (MKT == 1) & (np.abs(C / prevC - 1) <= 0.105)
        CHG[m3] = C[m3] - prevC[m3]
        if mode == 'main_plus':
            m4 = z & ~inx & (MKT == 1) & after
            CHG[m4] = C[m4] - prevC[m4]
        log['n_fixed'] = int(m.sum() + m3.sum())
        return CHG
    return fix


out = {}
for mode in ('orig', 'main', 'main_plus', 'close'):
    log = {}
    g = {'__name__': 'audit_list_partial', '__FIX__': make_fix(mode, log)}
    exec(compile(body, 'audit_list.py(partial)', 'exec'), g)
    B = g['B']; RET20 = g['RET20']; t = g['t']; ci = g['ci']; dist = g['dist']
    top = B['top20']
    out[mode] = dict(n_ma=B['n_ma'], top20=top, f1=B['f1'],
                     rows=[dict(code=c, rank=i + 1, ret20=round(float(RET20[t, ci[c]]) * 100, 2), dist_ma60=round(float(dist[ci[c]]) * 100, 2)) for i, c in enumerate(top)],
                     e0=[dict(code=c, rank=B['rank'].get(c)) for c in ('6213', '2355', '3311', '4938', '2480', '1802', '3416', '3037', '2492')],
                     log=log)
    print(mode, 'n_ma', B['n_ma'], 'F1', B['f1'], 'top20', top, flush=True)
json.dump(out, open(TMP / 'list_2026_11.json', 'w'), ensure_ascii=False, indent=1)
