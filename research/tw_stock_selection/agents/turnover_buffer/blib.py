"""turnover_buffer 共用：持股緩衝區選股（M2 成交）。

事前固定（見 FINDINGS「事前規則」）：
* 每個換股日 d，合格且 d+1 可交易的股票依分數排名（1 = 最高）。
* 舊持股（上一期實際持有）若仍合格、可交易、排名 ≤ K → 續抱。
* 不足 N 檔時，依排名從高到低補入其他股票；d+1 開盤漲停且未持有者視為買不到、由下一名遞補（與 mlib.select 相同）。
* K = N 時與 mlib.select（原版）逐期完全相同（程式內有斷言檢查）。
* 其餘（股池、成本、引擎、等權對手）完全沿用 momentum_family/mlib.py 與 exec_rebuild/exlib.py，不改原檔。
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'momentum_family'))
import mlib as M  # noqa: E402
L = M.L
pos = L.pos


def select_buf(score, el, dates, n, K):
    events, pools = {}, {}
    held = set(); nblk = 0; kept_n = []; new_names = []
    for d in dates[:-1]:
        i = pos(d); ex = i + 1
        okb = (el.loc[d] & score.loc[d].notna()).to_numpy()
        trad = np.isfinite(L.OPEN_RAW[ex]) & np.isfinite(L.ADJ_RAW[ex])
        ok = okb & trad
        if ok.sum() < 50: continue
        order = score.loc[d].to_numpy().astype(float).copy(); order[~ok] = np.nan
        cand = np.argsort(-np.nan_to_num(order, nan=-9e9))[:int(ok.sum())]
        rank = {int(j): r for r, j in enumerate(cand)}          # 0-based
        keep = sorted([j for j in held if j in rank and rank[j] < K], key=rank.get)
        pick = list(keep); ps = set(pick)
        for j in cand:
            if len(pick) >= n: break
            j = int(j)
            if j in ps: continue
            if M.GAP_UP[ex, j] and j not in held:
                nblk += 1; continue
            pick.append(j); ps.add(j)
        events[ex] = ('open', np.array(pick)); pools[ex] = ('open', np.where(ok)[0])
        if held:
            kept_n.append(len(set(pick) & held)); new_names.append(len(set(pick) - held) / n)
        held = set(pick)
    select_buf.nblk = nblk
    select_buf.mean_kept = float(np.mean(kept_n)) if kept_n else float('nan')
    select_buf.name_turnover = float(np.mean(new_names)) if new_names else float('nan')
    return events, pools


def same_events(e1, e2):
    if set(e1) != set(e2): return False
    return all(sorted(map(int, e1[t][1])) == sorted(map(int, e2[t][1])) for t in e1)
