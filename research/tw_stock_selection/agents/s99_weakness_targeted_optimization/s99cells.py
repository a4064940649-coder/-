#!/usr/bin/env python3
"""S99 變體定義（事前固定，見 PREREG.json）：同一期營收的月中第二次進場檢查（MID2）。只依賴 numpy／pandas；引擎物件（P、U、RT、S、ins15 遮蔽函式）由呼叫端傳入，
所以正式引擎（s99lib）與 2008 起連續引擎（cont2008.py）用的是同一份規則程式。

名稱＝池｜檢查日（6 格）：
  池 F＝訊號日當天的季線上 SURGE 前 10 名（名單固定，只是晚幾天再看一次「近 20 日漲幅 > 10%」）。
  池 R＝檢查日當天重排的季線上 SURGE 前 10 名（訊號日還在季線下、之後站上季線的也能進）。
  檢查日＝訊號日後第 5／第 10 個交易日，或兩天都看（5+10）。
SURGE 分數一律用訊號日 k 當時可見的那一份（含 15 日期限公司遮蔽）；檢查列 r＝k＋D 只多用列 ≤ r 的還原收盤（RET20、60 日均線）；第 r＋1 列開盤成交。"""
import numpy as np, pandas as pd

CELLS = ('F|5', 'F|10', 'F|5+10', 'R|5', 'R|10', 'R|5+10')
DAYS = {'5': (5,), '10': (10,), '5+10': (5, 10)}
DEPTH, MOM_LO = 10, 0.10                                   # 現行：季線上 SURGE 前 10 名、近 20 日漲幅 > 10%


def month_scores(P, RT, x, ins15):
    """訊號日 k 當時可見的當月 SURGE 分數（逐行同 s10lib.make_signal_s10）。"""
    m = [mm for mm in RT['S'].index if str(mm) == x['month']]
    assert len(m) == 1
    s = RT['S'].loc[m[0]].to_numpy()
    return ins15(P, m[0], x['k'], s)


def order_ma(P, U, s, row):
    """列 row 的「股池內、SURGE 有值、還原收盤 > 60 日均線」依 SURGE 由高到低（寫法同引擎）。"""
    ok = U[row] & np.isfinite(s) & P.F_MA60[row]
    return np.flatnonzero(ok)[np.argsort(-s[ok], kind='stable')]


def apply_cell(P, U, RT, S, cell, ins15):
    """cell＝'CUR' 或 '池|檢查日'。回傳 (S2, rec)。S2 多一個 mid_rows＝{月中檢查列: 營收月份}。
    每個訊號月（訊號日列 k、下一個訊號日列 kn）對每個檢查日 D：r＝k＋D；必須 r＋1 < kn（月中買進要在下一個訊號日之前成交）且 r＋1 < T，否則該次檢查略過。
      池 F：pool＝o_ma(k)[:10] 中列 r 仍在股池且收盤 > 60 日均線者；候選＝pool 中 RET20[r] > 10%、不在訊號日買進名單、同月較早檢查日沒加過者。
      池 R：pool＝o_ma(r)[:10]；候選條件同上。
    已持有或當天剛賣的由引擎自己略過（引擎原規則）；名額（10 檔）、每檔金額（衛星 ÷ 10）、開盤漲停不買、最小買進金額都照引擎。出場 X 不動。"""
    rec = []
    S2 = dict(S)
    if cell == 'CUR':
        S2['mid_rows'] = {}
        return S2, rec
    pool_kind, dk = cell.split('|'); days = DAYS[dk]; assert pool_kind in ('F', 'R')
    sig = S['sig'].copy(); rank = S['rank'].copy(); mid_rows = {}
    info = S['info']; ks = [x['k'] for x in info]
    for i, x in enumerate(info):
        k = x['k']; kn = ks[i + 1] if i + 1 < len(ks) else P.T + 10 ** 6
        s = month_scores(P, RT, x, ins15)
        tgt = set(int(j) for j in np.flatnonzero(S['sig'][k]))
        o_k = order_ma(P, U, s, k)
        chk = o_k[:DEPTH]; chk = set(int(j) for j in chk[P.RET20[k][chk] > MOM_LO][:10])
        assert chk == tgt, ('名單重建不符', x['month'])
        added_month = set()
        for D in days:
            r = k + D
            if r + 1 >= kn or r + 1 >= P.T:
                rec.append(dict(month=x['month'], k=int(k), D=D, row=int(r), skipped=True, added=[])); continue
            assert not sig[r].any(), ('檢查列已有訊號', x['month'], D)
            r20 = P.RET20[r]
            if pool_kind == 'F':
                pool = [int(j) for j in o_k[:DEPTH] if U[r, j] and P.F_MA60[r, j]]
            else:
                pool = [int(j) for j in order_ma(P, U, s, r)[:DEPTH]]
            cand = [j for j in pool if r20[j] > MOM_LO and j not in tgt and j not in added_month]
            for j in cand:
                sig[r, j] = True; rank[r, j] = s[j]
            added_month |= set(cand)
            if cand:
                mid_rows[int(r)] = x['month']
            rec.append(dict(month=x['month'], k=int(k), D=D, row=int(r), skipped=False, added=[str(P.COLS[j]) for j in cand]))
    S2['sig'] = sig; S2['rank'] = rank; S2['mid_rows'] = mid_rows
    return S2, rec


def mid_buy_counts(lg, mid_rows):
    """凍結格某一次模擬「月中實際買進」的檔數：{檢查列 r: 檔數}（交易明細的 sig_t＝r）。"""
    x = lg[lg['sig_t'].isin(list(mid_rows))]
    return {int(r): int(n) for r, n in x.groupby('sig_t').size().items()}


def random_overlay(P, U, S, counts, mid_rows, rng, pool='MA_MOM', avoid=None):
    """隨機對照：在同樣的檢查列 r 各加「同檔數」的隨機股票（counts＝mid_buy_counts 的結果），其餘與現行相同。
    pool='MA_MOM'：列 r 股池內、收盤 > 60 日均線、RET20[r] > 10%（同樣有動能，但不看 SURGE）；
    pool='ALLMA'：列 r 股池內、收盤 > 60 日均線（不看動能、不看 SURGE）。
    avoid＝{列 r: 該列不可抽的欄（例如當時已持有者，避免抽到被引擎略過而少買）}。rank 用隨機數（只決定同列內下單順序）。"""
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy(); mr = {}
    for r in sorted(counts):
        n = counts[r]
        base = U[r] & P.F_MA60[r] & np.isfinite(P.RET20[r])
        if pool == 'MA_MOM':
            base = base & (P.RET20[r] > MOM_LO)
        if avoid and r in avoid:
            base = base.copy(); base[list(avoid[r])] = False
        bj = np.flatnonzero(base)
        pick = rng.choice(bj, size=min(n, len(bj)), replace=False) if n and len(bj) else []
        for j in pick:
            sig[r, int(j)] = True; rank[r, int(j)] = float(rng.random())
        if len(pick):
            mr[int(r)] = mid_rows[r]
    S2['sig'] = sig; S2['rank'] = rank; S2['mid_rows'] = mr
    return S2


def fresh_from(S, first_month):
    """只保留營收月份 ≥ first_month 的進場訊號（含月中列）；出場 X 不動。"""
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first_month:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    mr = {}
    for r, m in S.get('mid_rows', {}).items():
        if pd.Period(m, 'M') < first_month:
            sig[r, :] = False; rank[r, :] = np.nan
        else:
            mr[r] = m
    S2['sig'] = sig; S2['rank'] = rank; S2['mid_rows'] = mr
    return S2
