#!/usr/bin/env python3
"""S16 共用程式庫：月營收驚奇 SURGE 股池加入上櫃。
在 s10lib（不改）之上：面板（官方日 K 本來就含上市＋上櫃，市場別逐日記在 P.MK：1＝上市、2＝上櫃）、創新板旗標、SURGE 訊號（S10 E0＝F1 進場＋掉出 L_MA 前 20 名才賣）、
引擎（s10lib.simulate_s10，不開任何出場掛鉤＝s9lib.simulate_s9 open）全部沿用。
本檔只新增：四個股池（CUR／ALL／OTC／ALL_O20）、四段空頭最大跌幅、最賺 5 筆占比、報酬貢獻（逐日損益 ÷ 前一日淨值）拆市場、5% 成交值上限觸發判斷。
中間檔放 scratchpad/s16tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S16 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s16tmp')
sys.path.insert(0, str(AG / 's10_exit_details'))
import s10lib                                              # noqa: E402
from s10lib import s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s10lib.S10 = S16; s9lib.S9 = S16; s8lib.S8 = S16; s7lib.S7 = S16; s6lib.S6 = S16; s5lib.S5 = S16
s4lib.S4 = S16; s3lib.S3 = S16; s2lib.S2 = S16; stlib.S1 = S16; cslib.TMP = S16   # 快取改放本題 scratchpad
SEGMENTS = {'2018': ('2018-01-01', '2018-12-31'), '2020-03': ('2020-01-01', '2020-06-30'),
            '2022': ('2022-01-01', '2022-12-31'), '2024-07~2025-04': ('2024-07-01', '2025-04-30')}
OTC_ADV_HI = 2e7          # 敏感度格 ALL_O20：上櫃股 60 日均成交值門檻提高到 2,000 萬（上市維持 1,000 萬）
CELLS = ('CUR', 'ALL', 'OTC', 'ALL_O20')


def load():
    S16.mkdir(parents=True, exist_ok=True)
    return s10lib.load()


def universes(P, INNOV):
    """四個股池（T×N 布林）。共同門檻＝stlib.liq：60 日均成交值（無成交日記 0、在日 K 內的日子滿 60 根，含上櫃時期）≥ 1,000 萬、收盤 ≥ 10、當日有成交、非 91xx；
    再排除創新板（當日名稱 -(KY)?創$）。市場別看「當日」P.MK（轉上市的股票轉換前算上櫃、之後算上市）。
    CUR＝只上市；ALL＝上市＋上櫃；OTC＝只上櫃；ALL_O20＝ALL 但上櫃股 60 日均成交值要 ≥ 2,000 萬。"""
    base = P.liq & ~INNOV
    tw, tp = P.MK == 1, P.MK == 2
    return {'CUR': base & tw, 'ALL': base & (tw | tp), 'OTC': base & tp,
            'ALL_O20': base & (tw | (tp & (np.nan_to_num(P.V60) >= OTC_ADV_HI)))}


def surge_signal(P, univ, RT, shift=0):
    """現行 SURGE 衛星（S10 E0）：s10lib.make_signal_s10 的 E0（同 s11lib.surge_signal）。"""
    S = s10lib.make_signal_s10(P, univ, RT, shift=shift)
    S2 = dict(S); S2['X'] = S['XV']['E0']; S2.pop('XV', None); S2.pop('RQ', None); S2['RQ'] = {}
    return S2


def seg_dd(nav):
    """四段空頭各自的最大跌幅（高點與低點都在段內；同 s11lib／s12lib.seg_dd）。"""
    out = {}
    for k, (a, b) in SEGMENTS.items():
        s = nav.loc[a:b].dropna()
        out[k] = float((s / s.cummax() - 1).min()) if len(s) > 2 else float('nan')
    return out


def top5_share(lg, nav, cap):
    """最賺 5 筆（已平倉，依損益金額）損益合計 ÷ 淨值增加（同 s12lib.top5_share）。"""
    x = lg[lg['why'] != 'open_end'].sort_values('pnl', ascending=False).head(5)
    return float(x['pnl'].sum() / (nav.iloc[-1] - cap))


def trade_contrib(P, lg, nav):
    """報酬貢獻口徑（同 S12 複製員）：每筆交易的逐日損益 ÷ 前一日總淨值後加總。
    逐日損益重建：進場日＝股數 × 收盤 − 買進成本；之後每天＝前一日市值 × 含息日報酬 R；出場日＝出場淨收入 − 前一日市值（開盤賣）。
    （底倉拉回時的部分賣出沒有逐日拆，金額已含在出場淨收入，逐筆合計＝損益金額，只是在持有期內的日期分配略有誤差。）
    回傳每筆交易在各期間（全期、2019 起、2019～2023、2024 起）的貢獻（小數）。"""
    navv = nav.reindex(P.IDX).to_numpy()
    d19 = int(np.searchsorted(P.IDX.values, np.datetime64('2019-01-01')))
    d24 = int(np.searchsorted(P.IDX.values, np.datetime64('2024-01-01')))
    out = np.zeros((len(lg), 4))
    for i, r in enumerate(lg.itertuples()):
        j, e, x = int(r.j), int(r.ent_t), int(r.exit_t)
        sh = float(r.shares)
        v = sh * P.CF[e, j]
        days = [e]; pnl = [v - float(r.cost)]
        last = x if r.why in ('open_end', 'delist') else x - 1
        for t in range(e + 1, last + 1):
            pv = v; v = v * (1 + P.R[t, j]); days.append(t); pnl.append(v - pv)
        if r.why == 'open_end':
            pnl[-1] += float(r.proceeds) - v
        elif r.why == 'delist':
            pnl[-1] += float(r.proceeds) - v
        else:
            days.append(x); pnl.append(float(r.proceeds) - v)
        days = np.asarray(days); pnl = np.asarray(pnl)
        c = pnl / navv[days - 1]
        out[i] = [c.sum(), c[days >= d19].sum(), c[(days >= d19) & (days < d24)].sum(), c[days >= d24].sum()]
    return pd.DataFrame(out, columns=['full', 'since2019', 'y2019_2023', 'since2024'], index=lg.index)


def nav_contrib(nav):
    """總淨值的報酬貢獻（逐日報酬加總，同口徑分母）。"""
    r = nav.pct_change().fillna(0.0)
    idx = r.index
    return {'full': float(r.sum()), 'since2019': float(r[idx >= '2019-01-01'].sum()),
            'y2019_2023': float(r[(idx >= '2019-01-01') & (idx < '2024-01-01')].sum()), 'since2024': float(r[idx >= '2024-01-01'].sum())}


def cap_bound(P, lg, cap_pct=0.05):
    """買進時 5% 成交值上限是否綁住：成交股數 ＝ int(5% × 當日成交值 ÷ 開盤)（2020-10-26 前取整張）。逐筆布林。"""
    odd_t = int(np.searchsorted(P.IDX.values, np.datetime64(cslib.ODD_DATE), side='left'))
    e = lg['ent_t'].to_numpy(int); j = lg['j'].to_numpy(int)
    o = P.O[e, j]; amt = P.AMT[e, j]
    mx = np.floor(cap_pct * amt / o)
    mx = np.where(e < odd_t, np.floor(mx / 1000) * 1000, mx)
    return np.isfinite(mx) & (lg['shares'].to_numpy(float) == mx)


# ---------------------------------------------------------------- 上櫃 2018-10 月營收補洞（PREREG design.data_patch）
MOPS_OTC_201810 = Path('/home/user/chijiun/tuc-6274-financial-forecast/data/raw/mops_monthly/t21sc03_107_10_0.html')   # 公開資訊觀測站上櫃 107 年 10 月營收彙總表（官方原始頁，他人 repo 的原始快取，只讀）
MOPS_OTC_201810_SHA = '1fb55c7bf25c6d17fd3e249a3733f7746c37f97f333c1e99ed544ba83d38f9ab'
PATCH_CSV = HERE / 'rev_patch_otc_201810.csv'


def parse_mops_t21(path, sha=None):
    """MOPS t21sc03 彙總表 → DataFrame(code, name, rev, prev_month, last_year)（千元）。只取四碼代號列。"""
    import re, hashlib
    b = Path(path).read_bytes()
    if sha is not None:
        assert hashlib.sha256(b).hexdigest() == sha, 'MOPS 原始頁 sha256 不符'
    s = b.decode('cp950', errors='replace')
    num = r'<t[dD] nowrap>\s*([-\d,\.]+)\s*</t[dD]>'
    pat = re.compile(r'<tr align=right><td align=center>(\d{4})</td><td align=left>([^<]*)</td>' + num + num + num)
    rows = [(c, n.strip(), float(a.replace(',', '')), float(p.replace(',', '')), float(l.replace(',', ''))) for c, n, a, p, l in pat.findall(s)]
    return pd.DataFrame(rows, columns=['code', 'name', 'rev', 'prev_month', 'last_year']).drop_duplicates('code')


def rev_tables_s16(P, patch=True):
    """s2lib.rev_tables 的逐行複製，只多一步：patch=True 時把 PATCH_CSV（2018-10 上櫃、tw_revenue.csv 沒有的代號）併進去。
    patch=False 與 s2lib.rev_tables 完全相同（驗收 A5）。"""
    r = pd.read_csv(stlib.REV_CSV, dtype={'code': str}, parse_dates=['month'])
    if patch:
        pt = pd.read_csv(PATCH_CSV, dtype={'code': str}, parse_dates=['month'])
        have = set(r.loc[r['month'] == pd.Timestamp('2018-10-01'), 'code'])
        pt = pt[~pt['code'].isin(have)]
        r = pd.concat([r, pt[['month', 'market', 'code', 'name', 'revenue_k_twd']]], ignore_index=True)
    r['pri'] = (r['market'] != 'sii').astype(int)
    r = r.sort_values(['month', 'code', 'pri']).drop_duplicates(['month', 'code'], keep='first')
    Rv = r.pivot(index='month', columns='code', values='revenue_k_twd').sort_index()
    Rv.index = pd.PeriodIndex(Rv.index, freq='M')
    Rv = Rv.reindex(pd.period_range(Rv.index.min(), Rv.index.max(), freq='M')).reindex(columns=P.COLS)
    Rv = Rv.where(Rv > 0)
    D = Rv - Rv.shift(12)
    D[(D.index.year == 2013)] = np.nan
    mu = D.shift(1).rolling(24, min_periods=24).mean()
    sd = D.shift(1).rolling(24, min_periods=24).std(ddof=1)
    S = ((D - mu) / sd.where(sd > 0)).replace([np.inf, -np.inf], np.nan)
    yoy = (Rv / Rv.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
    yoy3 = yoy.rolling(3, min_periods=3).mean()
    av = cslib.rev_avail_days(P, list(S.index), True)
    return dict(S=S, YOY=yoy, YOY3=yoy3, av=av)
