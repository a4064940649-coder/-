#!/usr/bin/env python3
"""S33 共用程式庫：George & Hwang（2004, JF）52 週高點接近度（PTH）能不能改進 SURGE 名單內的選股？
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎）之上，本檔只新增：
- 開 cslib.EXRIGHT_FIX＝True（除權息參考價修正；現行驗收 48.99／34.54／−32.11／381 筆已出場）。
- pth_panel(P)：PTH[t, j]＝還原收盤 G[t, j] ÷ max(G[t−251 … t, j])（含當日、過去 252 個交易日；P.nbar[t, j] ≥ 252 才有值，否則 NaN）。
  訊號列 k 只用到 k 當日（含）以前的收盤 → PIT（同 F1 的 RET20、F_MA60 在訊號日收盤判斷）。
- pth_row(k)：PTH 原始值與同期百分位 pPTH（股池 U[k] 內有值者排名，0～1，越大越接近高點）；缺值分數 PTHn＝0.5（中性，同 S29）。
- build_variant()：照 make_signal_s10 的同一個訊號列，依規則改當月買進名單（sig／rank）；出場 X（E0＝跌出 L_MA 前 20 名）完全不動，
  遞補、重排、換門檻都只在 L_MA 前 20 名內進行（同 S21／S29）。
- 隨機對照（rng）：凍結規則的 PTH 換成「當月 L_MA 前 20 名內 PTH 值的隨機重排」（分布與機制完全相同，只打亂對應）。
中間檔放 scratchpad/s33/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S33 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s33')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S33; s17lib.S17 = S33; s10lib.S10 = S33; s9lib.S9 = S33; s8lib.S8 = S33; s7lib.S7 = S33; s6lib.S6 = S33; s5lib.S5 = S33; s4lib.S4 = S33
s3lib.S3 = S33; s2lib.S2 = S33; stlib.S1 = S33; cslib.TMP = S33
cslib.EXRIGHT_FIX = True

TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
SEGMENTS = s17lib.SEGMENTS
fresh_from = s20lib.fresh_from
ACC = dict(since2019=0.4899, full=0.3454, maxdd=-0.3211, n_trades=381)   # EXRIGHT_FIX＝True 的現行驗收（已出場筆數）
WIN = 252
GATE = 0.90

RULES = [
    dict(id='RANK_PTH', kind='rank', score='PTH',
         desc='(a) 排序：L_MA 前 20 名內依「SURGE 名次＋PTH 名次」（等權；同分依 SURGE；PTH 缺值以同期百分位 0.5 代入）重排，取前 10 名為池，再要求 RET20 > 10%；買進順序照新排序'),
    dict(id='FLT_PTH', kind='filter_rel', score='PTH', frac=1 / 3,
         desc='(b) 篩選：F1 名單內依 PTH 排序，剔除最低的 floor(n／3) 檔（缺值以同期百分位 0.5 代入；同分剔除 SURGE 較低者）；不遞補，錢照引擎放 0050'),
    dict(id='FILL_PTH', kind='fill', score='PTH', min=GATE,
         desc='(c) 遞補：F1 不足 10 檔時，從 L_MA 第 11～20 名中 PTH ≥ 0.90 者（不看 RET20），依 PTH 由高到低遞補到 10 檔；PTH 缺值不遞補'),
    dict(id='GATE_PTH', kind='gate', score='PTH', thr=GATE,
         desc='(d) 換門檻：L_MA 前 10 名中以 PTH ≥ 0.90 取代 RET20 > 10% 當進場門檻（PTH 缺值〔上市未滿 252 根〕沿用 RET20 > 10%）；順序照 SURGE'),
]
CONTROL = dict(id='FILL_ALL', kind='fill', score=None,
               desc='對照（不參與選擇）：F1 不足 10 檔時，從 L_MA 第 11～20 名 RET20 > 10% 者依 SURGE 遞補，不看 PTH（同 S21／S29 FILL_ALL）')


def load():
    S33.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '未開 EXRIGHT_FIX'
    P.PTH = pth_panel(P)
    return P


def pth_panel(P):
    G = P.G.astype(np.float64)
    mx = pd.DataFrame(G).rolling(WIN, min_periods=WIN).max().to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        x = G / mx
    x = np.where((P.nbar >= WIN) & np.isfinite(x), x, np.nan)
    return x.astype(np.float32)


def surge(P):
    F = s17lib.rev_family(P, defs=['STD'])
    return dict(S=F['S']['STD'], av=F['av'])


class PTHS:
    def __init__(self, P, U):
        self.P, self.U = P, U
        self.cache = {}

    def row(self, k):
        if k in self.cache:
            return self.cache[k]
        x = self.P.PTH[k].astype(float)
        ok = self.U[k] & np.isfinite(x)
        pc = np.full(self.P.N, np.nan)
        if ok.sum() >= 30:
            pc[ok] = pd.Series(x[ok]).rank(pct=True, method='average').to_numpy()
        out = dict(PTH=x, pPTH=pc, PTHn=np.where(np.isfinite(pc), pc, 0.5))
        self.cache[k] = out
        return out


def row_parts(P, U, s, k, depth=10, mom_lo=0.10):
    """重現 make_signal_s10 在訊號列 k 的中間量：o_ma（L_MA 依 SURGE 排序）、F1 targets。"""
    fin = U[k] & np.isfinite(s)
    ok_ma = fin & P.F_MA60[k]
    o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
    r20 = P.RET20[k]
    pool = o_ma[:depth]
    tgt = pool[r20[pool] > mom_lo][:10]
    return fin, o_ma, tgt


def apply_rule_row(P, k, s, o_ma, tgt, sc, raw, rule):
    """單一訊號列套規則。sc＝分數（PTHn，越大越接近高點；None＝不看分數）；raw＝PTH 原始值（門檻用）。回傳 new（買進順序）與 rk。"""
    r20 = P.RET20[k]
    top20 = o_ma[:20]
    kind = rule['kind']
    rk = {int(j): float(s[j]) for j in top20}
    new = [int(j) for j in tgt]
    if kind == 'filter_rel':
        n = len(tgt); nrm = int(np.floor(n * rule['frac']))
        if nrm > 0:
            pos = np.arange(n)
            order = np.lexsort((-pos, sc[tgt]))             # 分數由低到高；同分 SURGE 較低（pos 大）者先剔
            drop = set(int(tgt[i]) for i in order[:nrm])
            new = [int(j) for j in tgt if int(j) not in drop]
    elif kind == 'gate':
        pool = o_ma[:10]
        new = [int(j) for j in pool if (raw[j] >= rule['thr'] if np.isfinite(raw[j]) else r20[j] > 0.10)][:10]
    elif kind == 'fill':
        if len(new) < 10:
            if sc is None:
                el = [int(j) for j in o_ma[10:20] if r20[j] > 0.10]
            else:
                el = [int(j) for j in o_ma[10:20] if np.isfinite(raw[j]) and raw[j] >= rule.get('min', -np.inf)]
                el = sorted(el, key=lambda j: (-raw[j], -s[j]))
            for j in el:
                if len(new) >= 10:
                    break
                new.append(j)
    elif kind == 'rank':
        n = len(top20)
        rs = np.arange(1, n + 1, dtype=float)
        rb = pd.Series(-sc[top20]).rank(method='average').to_numpy()
        key = rs + rb
        oi = np.lexsort((rs, key))
        order = top20[oi]
        new = [int(j) for j in order[:10] if r20[j] > 0.10][:10]
        for i0 in range(n):
            rk[int(top20[i0])] = float(1000.0 - key[i0] - 1e-3 * rs[i0])
    else:
        raise ValueError(kind)
    return new, rk


def build_variant(P, U, RT, S0, PS, rule, rng=None):
    """S0＝make_signal_s10 的結果（同一 shift）。rule＝None（現行）或 dict。
    rng 不為 None → 隨機對照：當月 L_MA 前 20 名的 PTH（原始值與分數一起）隨機重排後套同一規則。
    回傳 S1（sig／rank 已改；X、info 不動）與 log。"""
    S = RT['S']
    if rule is None:
        return dict(S0), []
    sig = np.zeros_like(S0['sig']); rank = np.full_like(S0['rank'], np.nan)
    log = []
    for x in S0['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        s = S.loc[m].to_numpy()
        fin, o_ma, tgt = row_parts(P, U, s, k)
        assert [str(P.COLS[j]) for j in tgt] == x['targets'], ('重現 F1 失敗', x['month'], k)
        if rule['score'] is None:
            sc = raw = None
        else:
            row = PS.row(k)
            sc, raw = row['PTHn'], row['PTH']
            if rng is not None:
                t20 = o_ma[:20]
                perm = rng.permutation(len(t20))
                sc = sc.copy(); raw = raw.copy()
                sc[t20] = sc[t20][perm]; raw[t20] = raw[t20][perm]
        new, rk = apply_rule_row(P, k, s, o_ma, tgt, sc, raw, rule)
        for j in new:
            sig[k, j] = True; rank[k, j] = rk.get(j, float(s[j]))
        log.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt], new=[str(P.COLS[j]) for j in new]))
    S1 = dict(S0); S1['sig'] = sig; S1['rank'] = rank
    return S1, log


def cand_table(P, U, RT, S0, PS):
    """每個訊號列 L_MA 前 20 名一列（grp＝F1／R11_20／POOL_NOMOM）：month、k、code、j、lma_rank、surge、ret20、PTH、pPTH。"""
    S = RT['S']; rows = []
    for x in S0['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        s = S.loc[m].to_numpy()
        fin, o_ma, tgt = row_parts(P, U, s, k)
        r = PS.row(k)
        tset = set(int(j) for j in tgt)
        for pos, j in enumerate(o_ma[:20]):
            grp = 'F1' if int(j) in tset else ('R11_20' if pos >= 10 else 'POOL_NOMOM')
            rows.append((x['month'], int(k), str(P.COLS[j]), int(j), grp, pos + 1, float(s[j]), float(P.RET20[k, j]), float(r['PTH'][j]), float(r['pPTH'][j])))
    return pd.DataFrame(rows, columns=['month', 'k', 'code', 'j', 'grp', 'lma_rank', 'surge', 'ret20', 'PTH', 'pPTH'])
