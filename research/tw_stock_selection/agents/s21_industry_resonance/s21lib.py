#!/usr/bin/env python3
"""S21 共用程式庫：產業共振——同 PIT 產業很多公司一起營收爆發時，SURGE 選股會不會更準？
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎）之上，本檔只新增：
- load_industry_pit()：讀 o_lc_sector_rotation_pit/raw/industry_pit.csv（多份證交所快照逐段生效；只讀，不複製）；
  ind_at(P, k)：訊號日 P.IDX[k] 當時的產業（start ≤ 日 < end；end 空白＝至今）。「未分類」或代碼不在檔內 → 缺值（-1）。
- breadth_row()：每個訊號列 k（營收月份 m）：股池 U[k] 且 SURGE 有值的股票，依當時產業分組，
  n_g＝有 SURGE 者家數、s_g＝其中 SURGE ≥ 2 家數；每檔股票的「同業共振」
  PEER＝s_g − 1[自己 SURGE ≥ 2]（不含自己的同業 SURGE ≥ 2 家數）、
  BRX＝PEER ÷ (n_g − 1)（不含自己的同業 SURGE ≥ 2 比例；n_g < 5 → 缺值）。
- build_variant()：照 s10lib.make_signal_s10 的同一個訊號列，依規則改當月買進名單（sig／rank）；出場 X（E0＝跌出 L_MA 前 20 名）完全不動。
  為了不改出場，遞補與重排都只在 L_MA 前 20 名內進行。
中間檔放 scratchpad/s21/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S21 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s21')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S21; s17lib.S17 = S21; s10lib.S10 = S21; s9lib.S9 = S21; s8lib.S8 = S21; s7lib.S7 = S21; s6lib.S6 = S21; s5lib.S5 = S21; s4lib.S4 = S21
s3lib.S3 = S21; s2lib.S2 = S21; stlib.S1 = S21; cslib.TMP = S21

IND_CSV = AG / 'o_lc_sector_rotation_pit' / 'raw' / 'industry_pit.csv'
TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
SEGMENTS = s17lib.SEGMENTS
fresh_from = s20lib.fresh_from
MISSING = {'未分類', '', 'nan'}
SURGE_HI = 2.0
MIN_N = 5

# 事前登記的規則（PREREG.json 內逐字記錄；candidates 參與訓練期選擇，control 只描述）
RULES = [
    dict(id='FLT_P2', kind='filter', res=('PEER', 2), missing='keep',
         desc='篩選：F1 名單中只買「同 PIT 產業（不含自己）至少還有 2 家 SURGE ≥ 2」者；空出名額不遞補，錢照引擎放 0050'),
    dict(id='FLT_REL', kind='filter', res=('REL', 0.0), missing='keep',
         desc='篩選：F1 名單中只買「同業 SURGE 廣度（不含自己，同業有 SURGE 者 ≥ 5 家）≥ 全股池廣度（不含自己）」者；同業不足 5 家（缺值）保留；不遞補'),
    dict(id='FILL_P2', kind='fill', res=('PEER', 2),
         desc='遞補：F1 不足 10 檔時，從 L_MA 第 11～20 名中 RET20 > 10% 且同業至少 2 家 SURGE ≥ 2 者，依 SURGE 由高到低遞補到 10 檔'),
    dict(id='SWAP_P2', kind='swap', res=('PEER', 2), missing='keep',
         desc='換股：先照 FLT_P2 剔除 F1 中不共振者，再照 FILL_P2 從 L_MA 第 11～20 名共振者遞補到 10 檔'),
    dict(id='RANK_BR', kind='rank', w_s=1.0, w_b=1.0,
         desc='排序加分：L_MA 前 20 名內，依「SURGE 名次＋同業廣度 BRX 名次」（等權；BRX 缺值當最差；同分依 SURGE）重排，取前 10 名為池，再要求 RET20 > 10%；買進順序照新排序'),
]
CONTROL = dict(id='FILL_ALL', kind='fill', res=None,
               desc='對照（不參與選擇）：F1 不足 10 檔時，從 L_MA 第 11～20 名 RET20 > 10% 者依 SURGE 遞補，不看共振；用來分離「遞補本身」與「共振」')


def load():
    S21.mkdir(parents=True, exist_ok=True)
    return s20lib.load()


def surge(P):
    """STD SURGE（s17lib.rev_family，σ > 0）與可用日 av：與 s20 train/validate 用的 RT 相同。"""
    F = s17lib.rev_family(P, defs=['STD'])
    return dict(S=F['S']['STD'], av=F['av'])


def load_industry_pit():
    df = pd.read_csv(IND_CSV, dtype=str, keep_default_na=False)
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'].replace('', '2100-01-01'))
    df['industry'] = df['industry'].str.strip()
    return df


def industry_matrix(P, ks):
    """回傳 dict(k → 長度 N 的 int 陣列，產業代碼；-1＝缺值) 與代碼表。只用訊號日當時有效的那一段。"""
    df = load_industry_pit()
    names = sorted(set(df['industry']) - MISSING)
    nid = {n: i for i, n in enumerate(names)}
    rows = {c: g[['start', 'end', 'industry']].to_numpy() for c, g in df.groupby('code')}
    out = {}
    for k in ks:
        d = P.IDX[k]
        a = np.full(P.N, -1, int)
        for j, c in enumerate(P.COLS):
            r = rows.get(str(c))
            if r is None:
                continue
            for s, e, ind in r:
                if s <= d < e:
                    a[j] = nid.get(ind, -1)
                    break
        out[k] = a
    return out, names


def breadth_row(s, fin, ind):
    """單一訊號列：s＝SURGE（長度 N）、fin＝股池且 SURGE 有值、ind＝產業代碼。
    回傳 PEER、BRX、NG、REL（長度 N；缺值 NaN）。REL＝BRX − MKT，MKT＝全股池（不含自己）SURGE ≥ 2 比例（扣掉全市場同時爆發，例如農曆年基期）。"""
    N = len(s)
    PEER = np.full(N, np.nan); BRX = np.full(N, np.nan); NG = np.full(N, np.nan); REL = np.full(N, np.nan)
    hi = fin & (np.nan_to_num(s, nan=-np.inf) >= SURGE_HI)
    ok = fin & (ind >= 0)
    if not ok.any():
        return PEER, BRX, NG, REL
    ng = np.bincount(ind[ok], minlength=ind.max() + 1)
    sg = np.bincount(ind[ok & hi], minlength=ind.max() + 1)
    jj = np.flatnonzero(ind >= 0)
    n = ng[ind[jj]].astype(float); sv = sg[ind[jj]].astype(float)
    selfhi = (hi[jj] & ok[jj]).astype(float)
    selfin = ok[jj].astype(float)
    peer = sv - selfhi
    denom = n - selfin
    NG[jj] = n
    PEER[jj] = peer
    with np.errstate(invalid='ignore', divide='ignore'):
        BRX[jj] = np.where((n >= MIN_N) & (denom > 0), peer / denom, np.nan)
        tot_hi = float(hi.sum()); tot_fin = float(fin.sum())
        mkt = (tot_hi - (hi[jj]).astype(float)) / (tot_fin - fin[jj].astype(float))
        REL[jj] = BRX[jj] - mkt
    return PEER, BRX, NG, REL


def row_parts(P, U, s, k, depth=10, mom_lo=0.10):
    """重現 make_signal_s10 在訊號列 k 的中間量：o_ma（L_MA 依 SURGE 排序）、pool、F1 targets。"""
    fin = U[k] & np.isfinite(s)
    ok_ma = fin & P.F_MA60[k]
    o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
    r20 = P.RET20[k]
    pool = o_ma[:depth]
    tgt = pool[r20[pool] > mom_lo][:10]
    return fin, o_ma, tgt


def resonant(j, PEER, BRX, REL, rule):
    """共振判定（遞補、篩選用）。rule['res']＝('PEER', 2)：同業（不含自己）SURGE ≥ 2 至少 2 家；('REL', 0)：同業廣度 ≥ 全市場廣度。缺值 → None。"""
    f, thr = rule['res']
    v = {'PEER': PEER, 'BRX': BRX, 'REL': REL}[f][j]
    if not np.isfinite(v):
        return None
    return bool(v >= thr)


def build_variant(P, U, RT, S0, IND, rule):
    """S0＝make_signal_s10 的結果（同一 shift）。rule＝None（現行）或 dict(kind, ...)：
    kind='filter'：F1 中只買共振者（缺值依 rule['missing']：'keep' 保留）；空出名額不遞補。
    kind='fill'  ：F1 不足 10 檔時，從 L_MA 第 11～20 名、RET20 > 10%、共振者（rule['res'] 為 None → 不看共振）依 SURGE 順序遞補到 10 檔。
    kind='swap'  ：先 filter，再照 fill 遞補到「原 F1 檔數＋原本就可遞補的空位」＝10 檔上限。
    kind='rank'  ：L_MA 前 20 名內，以 key＝w_s × SURGE 名次 ＋ w_b × BRX 名次（名次 1＝最好；BRX 缺值當最差）排序，取前 10 為池，再 RET20 > 10%。
    回傳 S1（sig／rank 已改；X、info 不動）與 log（每列：month, k, orig, new）。"""
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
        PEER, BRX, _, REL = breadth_row(s, fin, IND[k])
        r20 = P.RET20[k]
        top20 = o_ma[:20]
        kind = rule['kind']
        rk = {int(j): float(s[j]) for j in top20}
        if kind in ('filter', 'swap'):
            new = []
            for j in tgt:
                r = resonant(j, PEER, BRX, REL, rule)
                if r is None:
                    r = rule.get('missing', 'keep') == 'keep'
                if r:
                    new.append(int(j))
        else:
            new = [int(j) for j in tgt]
        if kind in ('fill', 'swap'):
            for j in o_ma[10:20]:
                if len(new) >= 10:
                    break
                if not (r20[j] > 0.10):
                    continue
                if rule.get('res') is not None and not resonant(j, PEER, BRX, REL, rule):   # 缺值（None）也不遞補
                    continue
                new.append(int(j))
        if kind == 'rank':
            n = len(top20)
            rs = np.arange(1, n + 1, dtype=float)                       # SURGE 名次（o_ma 已依 SURGE 由大到小）
            b = np.array([BRX[j] if np.isfinite(BRX[j]) else -np.inf for j in top20])
            rb = pd.Series(-b).rank(method='average').to_numpy()       # BRX 越大名次越前
            key = rule['w_s'] * rs + rule['w_b'] * rb
            order = top20[np.lexsort((rs, key))]
            pool = order[:10]
            new = [int(j) for j in pool if r20[j] > 0.10][:10]
            for i, j in enumerate(order):
                i0 = int(np.flatnonzero(top20 == j)[0]); rk[int(j)] = float(1000.0 - key[i0] - 1e-3 * rs[i0])   # 買進順序用 key（越小越先買；同分依 SURGE）
        for j in new:
            sig[k, j] = True; rank[k, j] = rk.get(j, float(s[j]))
        log.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt], new=[str(P.COLS[j]) for j in new]))
    S1 = dict(S0); S1['sig'] = sig; S1['rank'] = rank
    return S1, log


def f1_breadth_table(P, U, RT, S0, IND):
    """每個 F1 目標一列＋L_MA 第 11～20 名一列（事前描述與事後描述用）：month、k、code、grp（F1／R11_20）、surge、ret20、PEER、BRX、NG、ind。"""
    S = RT['S']; rows = []
    for x in S0['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        s = S.loc[m].to_numpy()
        fin, o_ma, tgt = row_parts(P, U, s, k)
        PEER, BRX, NG, REL = breadth_row(s, fin, IND[k])
        tset = set(int(j) for j in tgt)
        for pos, j in enumerate(o_ma[:20]):
            grp = 'F1' if int(j) in tset else ('R11_20' if pos >= 10 else 'POOL_NOMOM')
            rows.append((x['month'], int(k), str(P.COLS[j]), int(j), grp, pos + 1, float(s[j]), float(P.RET20[k, j]), float(PEER[j]), float(BRX[j]), float(NG[j]), float(REL[j]), int(IND[k][j])))
    return pd.DataFrame(rows, columns=['month', 'k', 'code', 'j', 'grp', 'lma_rank', 'surge', 'ret20', 'PEER', 'BRX', 'NG', 'REL', 'ind'])
