#!/usr/bin/env python3
"""S25 獨立複製：共用函式庫（複製員自寫）。
只依 s25 的 SPEC.json／PREREG.json 文字與 FINDINGS.md 條件；未讀、未 import、未執行 s25lib.py／prep.py／describe.py／train.py／validate.py／posthoc.py。
引擎沿用研究中心共用：s20lib → s17lib → s10lib（預設旗標；make_signal_s10、simulate_s10、e50_fix_suspension、seg_dd）。
PIT 產業：o_lc_sector_rotation_pit/raw/industry_pit.csv（只讀；start ≤ 日 < end，『未分類』與空白當缺值）。
PIT 市值：o_lc_sector_rotation_pit/data/shares_full.parquet × 收盤，股數落後 63 交易日（SPEC 寫「同 s24 build.py」）。
中間檔放 scratchpad/s25rep/（結束刪除）。一律 python3 -I -B。"""
import sys, math, gc
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
REP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s25rep')
sys.path.insert(0, str(AG / 's20_persistent_growth'))
import s20lib                                              # noqa: E402
from s20lib import s17lib, s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s20lib.S20 = REP; s17lib.S17 = REP; s10lib.S10 = REP; s9lib.S9 = REP; s8lib.S8 = REP; s7lib.S7 = REP; s6lib.S6 = REP; s5lib.S5 = REP; s4lib.S4 = REP
s3lib.S3 = REP; s2lib.S2 = REP; stlib.S1 = REP; cslib.TMP = REP
IND_CSV = AG / 'o_lc_sector_rotation_pit' / 'raw' / 'industry_pit.csv'
SHARES = AG / 'o_lc_sector_rotation_pit' / 'data' / 'shares_full.parquet'
MISSING = {'未分類', '', 'nan'}
TRAIN_LAST = pd.Period('2021-11', 'M')                     # 營收月份（訊號日 2016-02～2021-12）
VALID_FIRST = pd.Period('2021-12', 'M')                    # 營收月份（訊號日 2022-01 起）


def load_panel():
    """面板＋策略股池 U（只上市、流動性、非創新板）＋ STD SURGE（RT）＋ 0050 停牌修正。"""
    REP.mkdir(parents=True, exist_ok=True)
    P = s20lib.load()
    INNOV, _ = s4lib.innov_panel(P)
    U = stlib.universe(P, 'twse') & ~INNOV
    F = s17lib.rev_family(P, defs=['STD'])
    RT = dict(S=F['S']['STD'], av=F['av'])
    del F
    fix = s17lib.e50_fix_suspension(P)
    gc.collect()
    return P, U, INNOV, RT, fix


def mcap_panel(P):
    """PIT 市值：股數（ffill 到交易日）× 收盤；股數落後 63 交易日：MC[t]＝(股數[t−63]×收盤[t−63])×G[t]/G[t−63]，缺值時用當日。"""
    sh = pd.read_parquet(SHARES)
    sh = sh.reindex(sh.index.union(P.IDX)).ffill().reindex(P.IDX).reindex(columns=P.COLS).to_numpy(np.float64)
    now = sh * P.CF
    LAG = 63
    MC = np.full((P.T, P.N), np.nan)
    MC[LAG:] = now[:-LAG] * (P.G[LAG:] / P.G[:-LAG])
    MC = np.where(np.isfinite(MC), MC, now)
    del sh, now
    gc.collect()
    return MC


def industry_rows(P, rows):
    """回傳 (dict(row → 長度 N 的產業代碼陣列；-1＝缺值), 名稱表)。日期 d＝P.IDX[row]，取 start ≤ d < end 的段。"""
    df = pd.read_csv(IND_CSV, dtype=str, keep_default_na=False)
    df['industry'] = df['industry'].str.strip()
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'].replace('', '2100-01-01'))
    names = sorted(set(df['industry']) - MISSING)
    nid = {n: i for i, n in enumerate(names)}
    seg = {c: list(zip(g['start'], g['end'], g['industry'])) for c, g in df.groupby('code')}
    cols = [str(c) for c in P.COLS]
    out = {}
    for r in rows:
        d = P.IDX[r]
        a = np.full(P.N, -1, int)
        for j, c in enumerate(cols):
            for s, e, ind in seg.get(c, ()):
                if s <= d < e:
                    a[j] = nid.get(ind, -1)
                    break
        out[int(r)] = a
    return out, names


def month_end_rows(P, first='2015-01'):
    idx = pd.DatetimeIndex(P.IDX)
    me = pd.Series(np.arange(P.T), index=idx).groupby(idx.to_period('M')).max()
    return me[me.index >= pd.Period(first, 'M')]


# ---------------------------------------------------------------- B：規則
def sig_month_of(S0):
    """make_signal_s10 的 info → 每列 (營收月份 Period, 訊號列 k, 訊號日的日曆月份 Period, F1 目標欄位索引 list)。"""
    out = []
    for x in S0['info']:
        out.append((pd.Period(x['month'], 'M'), int(x['k']), pd.Period(x['sig_date'][:7], 'M'), x['targets']))
    return out


def apply_rule(P, S0, trig_rows, action):
    """trig_rows：要觸發的訊號列 k 集合。stop＝整列不買；half＝只留 F1（SURGE 排序）前 ceil(n/2) 檔。回傳新 S 與少買檔數。"""
    S2 = dict(S0)
    sig = S0['sig'].copy(); rank = S0['rank'].copy()
    dropped = 0
    for x in S0['info']:
        k = int(x['k'])
        if k not in trig_rows:
            continue
        tg = [P.ci[c] for c in x['targets']]                 # targets 已依 SURGE 由高到低
        drop = tg[int(math.ceil(len(tg) / 2)):] if action == 'half' else tg
        dropped += len(drop)
        if drop:
            sig[k, drop] = False; rank[k, drop] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2, dropped


def fresh(S, first=VALID_FIRST):
    S2 = dict(S); sig = S['sig'].copy(); rank = S['rank'].copy()
    for x in S['info']:
        if pd.Period(x['month'], 'M') < first:
            sig[x['k'], :] = False; rank[x['k'], :] = np.nan
    S2['sig'] = sig; S2['rank'] = rank
    return S2
