#!/usr/bin/env python3
"""S29 共用程式庫：課程 QMV（Quality 品質、Momentum 動能、Valuation 估值）能不能改進 SURGE 的選股排序？
在 s20lib（不改；其下 s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎）之上，本檔只新增：
- 開 cslib.EXRIGHT_FIX＝True（除權息參考價修正；現行驗收 48.99／34.54／−32.11／381 筆）。
- fund_tables()：季報（o_pead_margin_surprise_low_attention/data/fin_long.parquet 單季 XBRL＋avail_dates.parquet 市場可得日 A；A 缺 → 法定期限），
  每季 TTM 歸屬母公司淨利（連續 4 季都有值）、最近一季歸屬母公司權益。只讀、不複製。
- qmv_row(P, k)：訊號列 k（訊號日 d＝P.IDX[k]）的三個原始值與同期百分位（股池 U[k] 內、有值者，0～1，越大越好）：
  Q＝ROE＝TTM 淨利 ÷ 最近一季權益（取 A ≤ d 的最新一季；權益 ≤ 0 → 缺值）；
  M＝12-1 個股動能＝G[k−21] ÷ G[k−252] − 1（含息累積指數；跳過最近約一個月；k−252 時要已掛牌）；
  V＝E/P＝TTM 淨利 ÷ PIT 市值（股數落後 63 交易日，同 s24／s25 mcap_panel；虧損 → 負值、排最貴）。
  QMV＝三個百分位的平均（缺值的分項以 0.5 中性值代入）。
- build_variant()：照 make_signal_s10 的同一個訊號列，依規則改當月買進名單（sig／rank）；出場 X（E0＝跌出 L_MA 前 20 名）完全不動，
  遞補與重排都只在 L_MA 前 20 名內進行（同 S21）。
中間檔放 scratchpad/s29/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S29 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s29')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = S29; s17lib.S17 = S29; s10lib.S10 = S29; s9lib.S9 = S29; s8lib.S8 = S29; s7lib.S7 = S29; s6lib.S6 = S29; s5lib.S5 = S29; s4lib.S4 = S29
s3lib.S3 = S29; s2lib.S2 = S29; stlib.S1 = S29; cslib.TMP = S29
cslib.EXRIGHT_FIX = True

PEAD = AG / 'o_pead_margin_surprise_low_attention' / 'data'
SH_PARQ = AG / 'o_lc_sector_rotation_pit' / 'data' / 'shares_full.parquet'
IND_CSV = AG / 'o_lc_sector_rotation_pit' / 'raw' / 'industry_pit.csv'
TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
SEGMENTS = s17lib.SEGMENTS
fresh_from = s20lib.fresh_from
ACC = dict(since2019=0.4899, full=0.3454, maxdd=-0.3211, n_trades=381)   # EXRIGHT_FIX＝True 的現行驗收

# 事前登記的規則（PREREG.json 逐字記錄；candidates 參與訓練期選擇，control 只描述）
RULES = [
    dict(id='RANK_QMV', kind='rank', score='QMV',
         desc='排序：L_MA 前 20 名內依「SURGE 名次＋QMV 名次」（等權；同分依 SURGE）重排，取前 10 名為池，再要求 RET20 > 10%；買進順序照新排序'),
    dict(id='FLT_Q', kind='filter_rel', score='Q', frac=1 / 3,
         desc='篩選：F1 名單內依 Q（ROE 百分位）排序，剔除最差的 floor(n／3) 檔（Q 缺值以 0.5 代入；同分剔除 SURGE 較低者）；不遞補，錢照引擎放 0050'),
    dict(id='FILL_QMV', kind='fill', score='QMV', min=0.5,
         desc='遞補：F1 不足 10 檔時，從 L_MA 第 11～20 名中 RET20 > 10% 且 QMV ≥ 0.5（三分項同期百分位平均在中位數以上）者，依 QMV 由高到低遞補到 10 檔'),
    dict(id='FLT_V', kind='filter_abs', score='V', thr=0.2,
         desc='篩選（課程 GARP：成長不能買太貴）：F1 名單內剔除 V（E/P 同期股池百分位）< 0.2（全市場最貴 1/5，含虧損股）者；V 缺值保留；不遞補'),
    dict(id='RANK_QM', kind='rank', score='QM',
         desc='排序：同 RANK_QMV，但分數只用 Q 與 M 兩個百分位的平均（不含估值）'),
]
CONTROL = dict(id='FILL_ALL', kind='fill', score=None,
               desc='對照（不參與選擇）：F1 不足 10 檔時，從 L_MA 第 11～20 名 RET20 > 10% 者依 SURGE 遞補，不看 QMV（同 S21 FILL_ALL）')


def load():
    S29.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '未開 EXRIGHT_FIX'
    return P


def surge(P):
    F = s17lib.rev_family(P, defs=['STD'])
    return dict(S=F['S']['STD'], av=F['av'])


def mcap_panel(P, lag=63):
    """同 s24 build.py／s25lib.mcap_panel：股數（shares_full）× 收盤，股數落後 lag 交易日；(T, N) float32。"""
    sh = pd.read_parquet(SH_PARQ)
    sh = sh.reindex(sh.index.union(P.IDX)).ffill().reindex(P.IDX).reindex(columns=P.COLS).to_numpy(np.float64)
    now = sh * P.CF
    MC = np.full((P.T, P.N), np.nan)
    MC[lag:] = now[:-lag] * (P.G[lag:] / P.G[:-lag])
    MC = np.where(np.isfinite(MC), MC, now)
    return MC.astype(np.float32)


def fund_tables():
    """季報長表：code、kq、avail（A，缺 → 法定期限）、ttm（TTM 歸屬母公司淨利，連續 4 季）、eq（歸屬母公司權益）、gp_ttm、rev_ttm、fin（金融保險格式）。"""
    f = pd.read_parquet(PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'industry', 'nip_s', 'equity_parent', 'gp_s', 'rev_s'])
    f = f[f['code'].str.fullmatch(r'[1-9]\d{3}') & ~f['code'].str.startswith('91')].copy()
    assert not f.duplicated(['code', 'y', 'q']).any()
    f['kq'] = f['y'] * 4 + f['q'] - 1
    a = pd.read_parquet(PEAD / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A'])
    f = f.merge(a, on=['code', 'y', 'q'], how='left')
    dl = {1: (5, 15), 2: (8, 14), 3: (11, 14)}
    f['deadline'] = [pd.Timestamp(y + (q == 4), *(dl[q] if q < 4 else (3, 31))) for y, q in zip(f['y'], f['q'])]
    f['avail'] = pd.to_datetime(f['A']).fillna(f['deadline'])
    f['avail_from_A'] = f['A'].notna()
    f = f.sort_values(['code', 'kq']).reset_index(drop=True)
    g = f.groupby('code')
    for src, dst in (('nip_s', 'ttm'), ('gp_s', 'gp_ttm'), ('rev_s', 'rev_ttm')):
        f[dst] = g[src].transform(lambda s: s.rolling(4, min_periods=4).sum())
    f['kq_lag3'] = g['kq'].shift(3)
    bad = (f['kq'] - f['kq_lag3']) != 3
    f.loc[bad, ['ttm', 'gp_ttm', 'rev_ttm']] = np.nan
    f['eq'] = f['equity_parent']
    f['fin'] = f['industry'].isin({'bd', 'basi', 'ins', 'fh', 'mim'})
    return f[['code', 'kq', 'avail', 'avail_from_A', 'ttm', 'eq', 'gp_ttm', 'rev_ttm', 'fin']]


class QMV:
    """每個訊號列的 Q、M、V（原始值與同期百分位）。快取在記憶體（dict k → dict）。"""

    def __init__(self, P, U):
        self.P, self.U = P, U
        self.F = fund_tables()
        self.F = self.F[self.F['code'].isin(set(P.COLS))]
        self.MC = mcap_panel(P)
        self.cache = {}

    def fund_at(self, d):
        x = self.F[self.F['avail'] <= d]
        last = x.groupby('code').last()                    # 依 kq 排序後最後一筆＝最近一季已可得
        ci = self.P.ci
        js = np.array([ci[c] for c in last.index])
        ttm = np.full(self.P.N, np.nan); eq = np.full(self.P.N, np.nan); lagq = np.full(self.P.N, np.nan)
        ttm[js] = last['ttm'].to_numpy(float); eq[js] = last['eq'].to_numpy(float)
        dk = d.year * 4 + (d.month - 1) // 3               # 訊號日所在季序號
        lagq[js] = dk - last['kq'].to_numpy(float)          # 落後幾季（描述用）
        return ttm, eq, lagq

    def row(self, k):
        if k in self.cache:
            return self.cache[k]
        P = self.P
        d = P.IDX[k]
        ttm, eq, lagq = self.fund_at(d)
        with np.errstate(invalid='ignore', divide='ignore'):
            q = np.where(eq > 0, ttm / eq, np.nan)
            if k >= 252:
                m = P.G[k - 21] / P.G[k - 252] - 1
                m = np.where((P.MK[k - 252] > 0) & np.isfinite(m), m, np.nan)
            else:
                m = np.full(P.N, np.nan)
            mc = self.MC[k].astype(float)
            v = np.where(np.isfinite(mc) & (mc > 0), ttm / mc, np.nan)
        u = self.U[k]
        out = dict(Q=q, M=m, V=v, lagq=lagq)
        for nm, x in (('Q', q), ('M', m), ('V', v)):
            ok = u & np.isfinite(x)
            pc = np.full(P.N, np.nan)
            if ok.sum() >= 30:
                pc[ok] = pd.Series(x[ok]).rank(pct=True, method='average').to_numpy()
            out['p' + nm] = pc
        nz = lambda a: np.where(np.isfinite(a), a, 0.5)
        out['QMV'] = (nz(out['pQ']) + nz(out['pM']) + nz(out['pV'])) / 3
        out['QM'] = (nz(out['pQ']) + nz(out['pM'])) / 2
        out['Qn'] = nz(out['pQ']); out['Mn'] = nz(out['pM']); out['Vn'] = nz(out['pV'])
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


SCORE_KEY = {'QMV': 'QMV', 'QM': 'QM', 'Q': 'Qn', 'M': 'Mn', 'V': 'Vn'}


def apply_rule_row(P, k, s, o_ma, tgt, sc, rule):
    """單一訊號列套規則。sc＝分數向量（長度 N，越大越好；None＝不看分數）。回傳 new（股票 index 清單，買進順序）與 rk（rank 值）。"""
    r20 = P.RET20[k]
    top20 = o_ma[:20]
    kind = rule['kind']
    rk = {int(j): float(s[j]) for j in top20}
    new = [int(j) for j in tgt]
    if kind == 'filter_rel':
        n = len(tgt); nrm = int(np.floor(n * rule['frac']))
        if nrm > 0:
            pos = np.arange(n)                              # tgt 依 SURGE 由高到低；同分剔除 SURGE 較低者（pos 大者）
            order = np.lexsort((-pos, sc[tgt]))             # 分數由低到高；同分 pos 大的先
            drop = set(int(tgt[i]) for i in order[:nrm])
            new = [int(j) for j in tgt if int(j) not in drop]
    elif kind == 'filter_abs':
        v = rule['raw'] if 'raw' in rule else None
        new = [int(j) for j in tgt if not (np.isfinite(v[j]) and v[j] < rule['thr'])]
    elif kind == 'fill':
        if len(new) < 10:
            el = [int(j) for j in o_ma[10:20] if r20[j] > 0.10]
            if sc is not None:
                el = [j for j in el if sc[j] >= rule.get('min', -np.inf)]
                el = sorted(el, key=lambda j: (-sc[j], -s[j]))
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


def build_variant(P, U, RT, S0, QM, rule, rng=None):
    """S0＝make_signal_s10 的結果（同一 shift）。rule＝None（現行）或 dict(kind, score, ...)。
    rng 不為 None → 隨機對照：分數換成每月獨立的均勻亂數（規則機制完全相同）。
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
        r = dict(rule)
        if r['score'] is None:
            sc = None
        elif rng is not None:
            sc = rng.random(P.N)
            if r['kind'] == 'filter_abs':
                r['raw'] = sc
        else:
            row = QM.row(k)
            sc = row[SCORE_KEY[r['score']]]
            if r['kind'] == 'filter_abs':
                r['raw'] = row['p' + r['score']]
        new, rk = apply_rule_row(P, k, s, o_ma, tgt, sc, r)
        for j in new:
            sig[k, j] = True; rank[k, j] = rk.get(j, float(s[j]))
        log.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt], new=[str(P.COLS[j]) for j in new]))
    S1 = dict(S0); S1['sig'] = sig; S1['rank'] = rank
    return S1, log


def cand_table(P, U, RT, S0, QM):
    """每個訊號列：L_MA 前 20 名一列（grp＝F1／R11_20／POOL_NOMOM）：month、k、code、j、lma_rank、surge、ret20、Q、M、V、pQ、pM、pV、QMV、QM、lagq。"""
    S = RT['S']; rows = []
    for x in S0['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        s = S.loc[m].to_numpy()
        fin, o_ma, tgt = row_parts(P, U, s, k)
        r = QM.row(k)
        tset = set(int(j) for j in tgt)
        for pos, j in enumerate(o_ma[:20]):
            grp = 'F1' if int(j) in tset else ('R11_20' if pos >= 10 else 'POOL_NOMOM')
            rows.append((x['month'], int(k), str(P.COLS[j]), int(j), grp, pos + 1, float(s[j]), float(P.RET20[k, j]),
                         *(float(r[c][j]) for c in ('Q', 'M', 'V', 'pQ', 'pM', 'pV', 'QMV', 'QM', 'lagq'))))
    return pd.DataFrame(rows, columns=['month', 'k', 'code', 'j', 'grp', 'lma_rank', 'surge', 'ret20', 'Q', 'M', 'V', 'pQ', 'pM', 'pV', 'QMV', 'QM', 'lagq'])
