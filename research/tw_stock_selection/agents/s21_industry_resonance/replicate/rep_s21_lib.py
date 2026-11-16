#!/usr/bin/env python3
"""S21 產業共振：獨立複製用函式庫（複製員自寫）。
只依 s21 的 PREREG.json 規則文字與 FINDINGS.md 條件；未讀、未 import s21lib.py／train.py／validate.py／posthoc.py。
引擎沿用研究中心共用：s20lib → s17lib → s10lib（make_signal_s10、simulate_s10、variant_S、e50_fix_suspension、seg_dd、top5_share）。
PIT 產業：o_lc_sector_rotation_pit/raw/industry_pit.csv（只讀）。中間檔放 scratchpad/s21rep/。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
REP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s21rep')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = REP; s17lib.S17 = REP; s10lib.S10 = REP; s9lib.S9 = REP; s8lib.S8 = REP; s7lib.S7 = REP; s6lib.S6 = REP; s5lib.S5 = REP; s4lib.S4 = REP
s3lib.S3 = REP; s2lib.S2 = REP; stlib.S1 = REP; cslib.TMP = REP
IND_CSV = AG / 'o_lc_sector_rotation_pit' / 'raw' / 'industry_pit.csv'
TRAIN_LAST = pd.Period('2021-11', 'M')
VALID_FIRST = pd.Period('2021-12', 'M')


def load():
    REP.mkdir(parents=True, exist_ok=True)
    return s20lib.load()


class PITInd:
    """訊號日當天有效的產業（start ≤ 日 < end；end 空＝至今）；「未分類」→ 缺值（-1）。"""
    def __init__(self, P):
        d = pd.read_csv(IND_CSV, dtype=str)
        d['start'] = pd.to_datetime(d['start']); d['end'] = pd.to_datetime(d['end'])
        d.loc[d['industry'] == '未分類', 'industry'] = np.nan
        self.names = sorted(d['industry'].dropna().unique())
        nid = {n: i for i, n in enumerate(self.names)}
        d['iid'] = d['industry'].map(nid).fillna(-1).astype(int)
        ci = {c: j for j, c in enumerate(map(str, P.COLS))}
        d = d[d['code'].isin(ci)]
        self.d = d; self.ci = ci; self.N = P.N; self.cache = {}

    def at(self, date):
        date = pd.Timestamp(date)
        if date in self.cache:
            return self.cache[date]
        a = np.full(self.N, -1, int)
        x = self.d[(self.d['start'] <= date) & (self.d['end'].isna() | (date < self.d['end']))]
        for c, i in zip(x['code'], x['iid']):
            a[self.ci[c]] = i
        self.cache[date] = a
        return a


def features(s, fin, ind, nmin_incl_self=5):
    """訊號列特徵（只對 fin＝股池且 SURGE 有值的股票）。
    PEER＝同產業（不含自己）SURGE ≥ 2 家數；BRX＝PEER ÷（同業有 SURGE 家數 − 1），同業有 SURGE 家數（含自己）< nmin → 缺值；
    MKT＝全股池（不含自己）SURGE ≥ 2 比例；REL＝BRX − MKT。產業缺值 → PEER、BRX、REL 缺值。"""
    N = len(s)
    hi = fin & (np.nan_to_num(s, nan=-np.inf) >= 2.0)
    PEER = np.full(N, np.nan); BRX = np.full(N, np.nan)
    okind = fin & (ind >= 0)
    ng = np.bincount(ind[okind], minlength=64)
    cg = np.bincount(ind[okind & hi], minlength=64)
    idx = np.flatnonzero(okind)
    PEER[idx] = cg[ind[idx]] - hi[idx]
    n_incl = ng[ind[idx]]
    with np.errstate(invalid='ignore', divide='ignore'):
        BRX[idx] = np.where(n_incl >= nmin_incl_self, PEER[idx] / (n_incl - 1), np.nan)
    nall = int(fin.sum()); call = int(hi.sum())
    MKT = np.full(N, np.nan)
    f = np.flatnonzero(fin)
    MKT[f] = (call - hi[f]) / (nall - 1)
    REL = BRX - MKT
    return dict(PEER=PEER, BRX=BRX, MKT=MKT, REL=REL, n_fin=nall, share_hi=call / nall if nall else np.nan)


def rank_desc(v):
    """由高到低名次（1＝最好），缺值最差；同值取平均名次。"""
    x = pd.Series(np.where(np.isfinite(v), v, -np.inf))
    return x.rank(ascending=False, method='average').to_numpy()


def build_variants(P, U, RT, PI, S0, shift, rules, depth=10, mom_lo=0.10, nsel=10, nmin=5, rank_method='average', rng=None, rand_spec=None):
    """自寫訊號迴圈（同 make_signal_s10 的 F1 定義），回傳 {規則 id: (sig, rank)} 與每列明細。
    S0＝同一位移的 make_signal_s10 結果（A1：每列 F1 必須與其 info['targets'] 相同）。
    rand_spec＝{k: (n_remove, n_add)}：隨機換股對照（每列隨機剔除 n_remove 檔 F1，從第 11～20 名 RET20>10% 者隨機遞補 n_add 檔）。"""
    S, av = RT['S'], RT['av']
    tgt_ref = {int(x['k']): x['targets'] for x in S0['info']}
    out = {r: (np.zeros((P.T, P.N), bool), np.full((P.T, P.N), np.nan)) for r in rules}
    rows = []
    COLS = np.asarray(P.COLS).astype(str)
    for m in S.index:
        k0 = av[m]
        if k0 >= P.T or k0 < 61:
            continue
        s = S.loc[m].to_numpy()
        if int((U[k0] & np.isfinite(s)).sum()) < 50:
            continue
        k = k0 + shift
        if k >= P.T:
            continue
        fin = U[k] & np.isfinite(s)
        ok_ma = fin & P.F_MA60[k]
        o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
        r20 = P.RET20[k]
        pool = o_ma[:depth]
        f1 = pool[r20[pool] > mom_lo][:nsel]
        assert list(COLS[f1]) == list(tgt_ref[int(k)]), ('A1 不符', m, shift)
        ind = PI.at(P.IDX[k])
        F = features(s, fin, ind, nmin)
        PEER, REL, BRX = F['PEER'], F['REL'], F['BRX']
        res_ok = lambda j: np.isfinite(PEER[j]) and PEER[j] >= 2          # 共振（遞補用：缺值不算共振）
        keep_p2 = lambda j: (not np.isfinite(PEER[j])) or PEER[j] >= 2    # 篩選用：缺值保留
        r1120 = [j for j in o_ma[depth:20] if r20[j] > mom_lo]
        rec = dict(month=str(m), k=int(k), f1=[int(j) for j in f1], r1120=[int(j) for j in r1120],
                   PEER={int(j): float(PEER[j]) for j in list(o_ma[:20])}, BRX={int(j): float(BRX[j]) for j in list(o_ma[:20])},
                   REL={int(j): float(REL[j]) for j in list(o_ma[:20])}, n_fin=F['n_fin'], share_hi=F['share_hi'],
                   ind={int(j): int(ind[j]) for j in list(o_ma[:20])})
        new = {}
        for r in rules:
            order = None
            if r == 'CUR':
                t = list(f1)
            elif r == 'PASSALL':
                t = [j for j in f1 if (not np.isfinite(PEER[j])) or PEER[j] >= -1]
            elif r == 'FLT_P2':
                t = [j for j in f1 if keep_p2(j)]
            elif r == 'FLT_REL':
                t = [j for j in f1 if (not np.isfinite(REL[j])) or REL[j] >= 0.0]
            elif r in ('FILL_P2', 'FILL_ALL', 'SWAP_P2'):
                t = [j for j in f1 if keep_p2(j)] if r == 'SWAP_P2' else list(f1)
                for j in r1120:
                    if len(t) >= nsel:
                        break
                    if r == 'FILL_ALL' or res_ok(j):
                        t.append(j)
            elif r == 'RANK_BR':
                top = o_ma[:20]
                rs = np.arange(1, len(top) + 1, dtype=float)
                rb = pd.Series(np.where(np.isfinite(BRX[top]), BRX[top], -np.inf)).rank(ascending=False, method=rank_method).to_numpy()
                sc = rs + rb
                o2 = np.lexsort((rs, sc))
                pl = top[o2][:depth]
                t = [j for j in pl if r20[j] > mom_lo][:nsel]
                order = t
            elif r == 'RAND':
                nr, na = rand_spec.get(int(k), (0, 0))
                f1l = list(f1)
                rm = set(rng.choice(f1l, size=nr, replace=False).tolist()) if nr > 0 else set()
                t = [j for j in f1l if j not in rm]
                pool2 = [j for j in r1120 if j not in t]
                na = min(na, len(pool2), nsel - len(t))
                if na > 0:
                    t += sorted(rng.choice(pool2, size=na, replace=False).tolist(), key=lambda j: -s[j])
            else:
                raise KeyError(r)
            sig, rank = out[r]
            t = list(t)
            sig[k, t] = True
            if order is None:
                rank[k, t] = s[t]
            else:
                rank[k, t] = np.arange(len(t), 0, -1, dtype=float)
            new[r] = [int(j) for j in t]
        rec['new'] = new
        rows.append(rec)
    return out, rows


def variant_S(S0, sig, rank):
    """把規則名單換進 S0（X、XV、info、sigrows 都沿用 E0）。"""
    S2 = dict(S0); S2['sig'] = sig; S2['rank'] = rank
    S3, ekw = s10lib.variant_S(S2, 'E0')
    return S3, ekw


def fresh(S, rows, first=VALID_FIRST):
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2
