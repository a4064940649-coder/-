#!/usr/bin/env python3
"""S28 共用程式庫：營收爆發但毛利率在惡化的公司，是不是 SURGE 的地雷？
在 s21lib（不改；其下 s20lib→s17lib→s10lib：面板、創新板、SURGE（STD，σ>0）、F1、E0、引擎）之上，本檔只新增：
- 除權息修正：import 後立刻設 cslib.EXRIGHT_FIX = True（2026-11-16 起的修正後引擎；驗收 48.99／34.54／−32.11／381 筆）。
- fin_quarters()：讀 o_pead_margin_surprise_low_attention/data/fin_long.parquet（MOPS XBRL 單季財報；只讀，不複製）：
  單季營收 rev_s、毛利 gp_s、營業利益 opi_s → 毛利率 GM、營業利益率 OPM、單季營收年增 RY、GM／OPM 年增（個百分點）。
  avail_dates.parquet 的市場可得日 A（上市 2017-04 起直接觀察，之前推斷）。
- pit_quarter()：訊號列 k（日期 d＝P.IDX[k]）每檔股票能用的「最近一季」：
  季序號 kq 可用 ⇔ 法定期限 deadline(kq) < d 且（A 缺值 或 A < d）；取可用者中最新的一季，往回最多看 1 季（逐檔，只處理晚交）。
  法定期限：證券交易法第 36 條（Q1 5/15、Q2 8/14、Q3 11/14、Q4 次年 3/31；金融保險業與 -KY 的 Q2 是 8/31——A10 訊號日都在 9/10 之後，不影響）。
- margin_row()：訊號列上每檔的 RY、dGM、dOPM、用到的季別、資料齡。
- flag()：「營收爆發但毛利率惡化」判定（事前登記，見 PREREG.json）。
- build_variant()：照 s10lib.make_signal_s10 的同一個訊號列，依規則改當月買進名單；出場 X（E0）完全不動。遞補只在 L_MA 第 11～20 名內。
中間檔放 scratchpad/s28/。一律 python3 -I -B。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S28 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s28')
sys.path.insert(0, str(AG / 's21_industry_resonance'))
import s21lib                                              # noqa: E402
from s21lib import s20lib, s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s21lib.S21 = S28; s20lib.S20 = S28; s17lib.S17 = S28; s10lib.S10 = S28; s9lib.S9 = S28; s8lib.S8 = S28; s7lib.S7 = S28; s6lib.S6 = S28; s5lib.S5 = S28; s4lib.S4 = S28
s3lib.S3 = S28; s2lib.S2 = S28; stlib.S1 = S28; cslib.TMP = S28
cslib.EXRIGHT_FIX = True                                   # 修正後引擎（題目指定）

PEAD = AG / 'o_pead_margin_surprise_low_attention' / 'data'
TRAIN_LAST_MONTH = s20lib.TRAIN_LAST_MONTH                 # 營收月份 ≤ 2021-11（訊號日 2016-02～2021-12）
VALID_FIRST_MONTH = s20lib.VALID_FIRST_MONTH               # 營收月份 ≥ 2021-12（訊號日 2022-01 起）
fresh_from = s20lib.fresh_from
row_parts = s21lib.row_parts
MAX_BACK = 1                                               # 最近一季還沒公告（晚交）時，只往回找 1 季

# 事前登記的規則（PREREG.json 內逐字記錄）
RULES = [
    dict(id='BAN_GM2', kind='filter', gm=2.0, opm=None,
         desc='禁買：F1 名單中，訊號日前已公告的最近一季「單季營收年增 > 0」且「毛利率年減 > 2 個百分點」者不買；空出名額不遞補，錢照引擎放 0050'),
    dict(id='BAN_GM5', kind='filter', gm=5.0, opm=None,
         desc='禁買：同 BAN_GM2，但門檻為毛利率年減 > 5 個百分點（X 在訓練期於 2、5 之間選）'),
    dict(id='BAN_GMOP', kind='filter', gm=2.0, opm=2.0,
         desc='禁買：最近一季「單季營收年增 > 0」且「毛利率年減 > 2 個百分點」且「營業利益率年減 > 2 個百分點」（獲利率惡化連營業利益都確認）者不買；不遞補'),
    dict(id='SWAP_GM2', kind='swap', gm=2.0, opm=None,
         desc='換股：先照 BAN_GM2 剔除，再從 L_MA 第 11～20 名中 RET20 > 10% 且未被 BAN_GM2 判定者，依 SURGE 由高到低遞補到 10 檔'),
]


def load():
    S28.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    assert P.exright_fix, '除權息修正沒有開'
    return P


def deadline(y, q):
    return {1: pd.Timestamp(y, 5, 15), 2: pd.Timestamp(y, 8, 14), 3: pd.Timestamp(y, 11, 14), 4: pd.Timestamp(y + 1, 3, 31)}[q]


def fin_quarters():
    """回傳 dict(GM, OPM, RY, DGM, DOPM, REV（DataFrame：index＝code，columns＝kq＝4y＋q−1）, A（dict (code,kq)→Timestamp）, ind)。
    GM＝gp_s ÷ rev_s（rev_s > 0 才算）；OPM＝opi_s ÷ rev_s；RY＝rev_s(kq) ÷ rev_s(kq−4) − 1（兩季 rev_s > 0）；
    DGM＝(GM(kq) − GM(kq−4)) × 100（個百分點）；DOPM 同。"""
    f = pd.read_parquet(PEAD / 'fin_long.parquet', columns=['code', 'y', 'q', 'industry', 'rev_s', 'gp_s', 'opi_s'])
    f = f[f['code'].str.fullmatch(r'[1-9]\d{3}')].copy()
    assert not f.duplicated(['code', 'y', 'q']).any()
    f['kq'] = f['y'] * 4 + f['q'] - 1
    piv = lambda c: f.pivot(index='code', columns='kq', values=c)
    REV = piv('rev_s'); ks = np.arange(REV.columns.min(), REV.columns.max() + 1)
    REV = REV.reindex(columns=ks); GP = piv('gp_s').reindex(index=REV.index, columns=ks); OP = piv('opi_s').reindex(index=REV.index, columns=ks)
    pos = REV.where(REV > 0)
    GM = GP / pos; OPM = OP / pos
    GM = GM.where(GM.abs() <= 5); OPM = OPM.where(OPM.abs() <= 5)          # 營收極小造成的離譜比率當缺值
    RY = pos / pos.shift(4, axis=1) - 1
    DGM = (GM - GM.shift(4, axis=1)) * 100
    DOPM = (OPM - OPM.shift(4, axis=1)) * 100
    a = pd.read_parquet(PEAD / 'avail_dates.parquet', columns=['code', 'y', 'q', 'A', 'A_method'])
    a['kq'] = a['y'] * 4 + a['q'] - 1
    A = {(c, k): d for c, k, d in zip(a['code'], a['kq'], a['A']) if pd.notna(d)}
    ind = f.sort_values('kq').groupby('code')['industry'].last()
    return dict(GM=GM, OPM=OPM, RY=RY, DGM=DGM, DOPM=DOPM, REV=REV, A=A, ind=ind)


def latest_kq_by_deadline(d):
    """日期 d 時法定期限已過（deadline < d）的最新季序號。"""
    y = d.year
    for kq in range(4 * y + 3, 4 * (y - 2), -1):
        if deadline(kq // 4, kq % 4 + 1) < d:
            return kq
    raise ValueError(d)


def margin_row(P, FQ, k, codes_j):
    """訊號列 k、欄位 codes_j（P 的欄號陣列）：每檔回傳 dict(kq, age_days, RY, DGM, DOPM)；kq＝-1 表示找不到可用季。
    PIT：kq 可用 ⇔ deadline(kq) < d 且（A 缺值或 A < d）；從法定期限已過的最新季往回最多 MAX_BACK 季，取第一個可用且 RY／DGM 至少一個有值的季。
    注意：往回時只看『可用』（已公告），不是挑有利的季——第一個已公告且有營收、毛利資料的季就用。"""
    d = P.IDX[k]
    k0 = latest_kq_by_deadline(d)
    out = {}
    DGM, DOPM, RY, A = FQ['DGM'], FQ['DOPM'], FQ['RY'], FQ['A']
    for j in codes_j:
        c = str(P.COLS[j])
        res = dict(kq=-1, age_days=np.nan, RY=np.nan, DGM=np.nan, DOPM=np.nan)
        if c in DGM.index:
            for kq in range(k0, k0 - MAX_BACK - 1, -1):
                av = A.get((c, kq))
                if av is not None and not (av < d):
                    continue                                   # 還沒公告
                if kq not in DGM.columns:
                    continue
                ry, dg, do = RY.at[c, kq], DGM.at[c, kq], DOPM.at[c, kq]
                if not (np.isfinite(ry) or np.isfinite(dg)):
                    continue                                   # 這一季沒有營收資料（多半是還沒上市／格式不同）
                y, q = kq // 4, kq % 4 + 1
                qe = pd.Timestamp(y, 3 * q, 1) + pd.offsets.MonthEnd(0)
                res = dict(kq=int(kq), age_days=float((d - qe).days), RY=float(ry), DGM=float(dg), DOPM=float(do))
                break
        out[int(j)] = res
    return out


def flag(m, rule):
    """「營收爆發但毛利率惡化」：最近一季單季營收年增 > 0 且 毛利率年減 > rule['gm'] 個百分點（且 rule['opm'] 有值時營業利益率年減 > rule['opm']）。
    任何需要的值缺 → False（保留，不禁買）。營收年減的季（衰退谷底季）一律不判定 → 避免擋到「由衰轉盛」。"""
    ry, dg, do = m['RY'], m['DGM'], m['DOPM']
    if not (np.isfinite(ry) and np.isfinite(dg)) or not (ry > 0):
        return False
    if not (dg < -rule['gm']):
        return False
    if rule.get('opm') is not None:
        if not np.isfinite(do) or not (do < -rule['opm']):
            return False
    return True


def build_variant(P, U, RT, S0, FQ, rule, cache=None):
    """S0＝make_signal_s10 的結果（同一 shift）。rule＝None（現行）或 dict(kind='filter'|'swap', gm, opm)。
    filter：F1 中被 flag 者不買；不遞補。swap：filter 後，從 L_MA 第 11～20 名 RET20 > 10% 且未被 flag 者依 SURGE 遞補到 10 檔。
    cache：dict，存每列的 margin_row（同一 shift 內共用）。回傳 S1（sig／rank 已改；X 不動）與 log。"""
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
        top20 = o_ma[:20]
        if cache is not None and k in cache:
            MR = cache[k]
        else:
            MR = margin_row(P, FQ, k, top20)
            if cache is not None:
                cache[k] = MR
        new = [int(j) for j in tgt if not flag(MR[int(j)], rule)]
        flagged = [str(P.COLS[j]) for j in tgt if flag(MR[int(j)], rule)]
        if rule['kind'] == 'swap':
            r20 = P.RET20[k]
            for j in o_ma[10:20]:
                if len(new) >= 10:
                    break
                if not (r20[j] > 0.10) or flag(MR[int(j)], rule):
                    continue
                new.append(int(j))
        for j in new:
            sig[k, j] = True; rank[k, j] = float(s[j])
        log.append(dict(month=x['month'], k=int(k), orig=[str(P.COLS[j]) for j in tgt], new=[str(P.COLS[j]) for j in new], flagged=flagged))
    S1 = dict(S0); S1['sig'] = sig; S1['rank'] = rank
    return S1, log


def f1_margin_table(P, U, RT, S0, FQ, G=None):
    """每個 F1 目標一列＋L_MA 第 11～20 名一列：month、k、code、j、grp、lma_rank、surge、ret20、kq、age_days、RY、DGM、DOPM、（G 有給時）MU、MS。"""
    S = RT['S']; rows = []
    for x in S0['info']:
        m = pd.Period(x['month'], 'M'); k = x['k']
        s = S.loc[m].to_numpy()
        fin, o_ma, tgt = row_parts(P, U, s, k)
        MR = margin_row(P, FQ, k, o_ma[:20])
        tset = set(int(j) for j in tgt)
        for pos, j in enumerate(o_ma[:20]):
            grp = 'F1' if int(j) in tset else ('R11_20' if pos >= 10 else 'POOL_NOMOM')
            r = MR[int(j)]
            c = str(P.COLS[j])
            mu = float(G['MUAMT'].at[m, c]) if G is not None else np.nan
            ms = float(G['MS'].at[m, c]) if G is not None else np.nan
            rows.append((x['month'], int(k), c, int(j), grp, pos + 1, float(s[j]), float(P.RET20[k, j]), r['kq'], r['age_days'], r['RY'], r['DGM'], r['DOPM'], mu, ms))
    return pd.DataFrame(rows, columns=['month', 'k', 'code', 'j', 'grp', 'lma_rank', 'surge', 'ret20', 'kq', 'age_days', 'RY', 'DGM', 'DOPM', 'MU', 'MS'])
