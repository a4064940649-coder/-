#!/usr/bin/env python3
"""S14 共用程式庫：公司與制度事件——選定事件「處置股（上市、第一次處置）」的事件衛星，與和 SURGE 合併。
在 s11lib／s10lib（都不改）之上：面板、創新板、SURGE 訊號（S10 E0）、引擎（s10lib.simulate_s10，不開任何出場掛鉤）、回撤分段與相關係數全部沿用；
本檔只新增：官方處置名單讀取（只讀 o_disposal_list_pit_audit_pkl18/data/DISP_OFFICIAL_v2.csv，sha256 8f97b304…）、處置鏈（延長規則）、事件訊號 make_signal_disp。
中間檔放 scratchpad/s14tmp/。規格見 PREREG.json。一律 python3 -I -B。"""
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
S14 = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s14tmp')
sys.path.insert(0, str(AG / 's11_pead_second_signal'))
import s11lib                                              # noqa: E402
from s11lib import s10lib, s9lib, s8lib, s7lib, s6lib, s5lib, s4lib, s3lib, s2lib, stlib, cslib, COSTS   # noqa: E402,F401
s11lib.S11 = S14; s10lib.S10 = S14; s9lib.S9 = S14; s8lib.S8 = S14; s7lib.S7 = S14; s6lib.S6 = S14; s5lib.S5 = S14
s4lib.S4 = S14; s3lib.S3 = S14; s2lib.S2 = S14; stlib.S1 = S14; cslib.TMP = S14   # 快取改放本題 scratchpad
SEGMENTS = s11lib.SEGMENTS
seg_dd, month_excess, corr_excess, surge_signal = s11lib.seg_dd, s11lib.month_excess, s11lib.corr_excess, s11lib.surge_signal
DISP_CSV = AG / 'o_disposal_list_pit_audit_pkl18' / 'data' / 'DISP_OFFICIAL_v2.csv'
DISP_SHA = '8f97b304411c6d013635ccc8fad9976f21f0fb4c9f0704bb0ee1bf678f052326'
EXTRA = AG / 'o_disposal_list_pit_audit_pkl18' / 'raw' / 'screener_extras'


def load():
    S14.mkdir(parents=True, exist_ok=True)
    return s10lib.load()


# ---------------------------------------------------------------- 官方處置名單
def disp_table():
    b = DISP_CSV.read_bytes()
    assert hashlib.sha256(b).hexdigest() == DISP_SHA, '處置名單 sha256 不符'
    d = pd.read_csv(DISP_CSV, dtype={'code': str})
    for c in ('announce', 'start', 'end'):
        d[c] = pd.to_datetime(d[c])
    return d


def disp_chains(P, D, entry='first'):
    """把官方處置名單變成「處置鏈」（未位移）。
    每筆記錄：s_row＝第一個 ≥ start 的面板交易日；k＝s_row − 1（訊號列＝公告日當天或之前最後一個交易日；買進在 s_row 開盤）；
              e_row＝最後一個 ≤ end 的面板交易日（處置最後一天；賣出在 e_row＋1 開盤）。
    新進場事件：entry='first' → kind=='regular' 且 nth==1（第一次處置）；entry='all' → kind=='regular'（第一次＋第二次）。
    延長：持有中（鏈的預定最後一天 e_row 之前或當天）公告的同檔任何處置記錄（不論 kind／nth），e_row 延到該記錄的 e_row（可連續延長）。
          公告日 ≤ 預定最後一天＝在預定賣出（隔天開盤）之前已公開，實盤做得到。被併入鏈的記錄不另當新事件。
    回傳 DataFrame：code, j, k, s_row, e_row, n_ext, nth0, announce0, start0, end_final。"""
    D = D[D['code'].isin(set(P.COLS))].copy()
    idx = P.IDX.values
    D['s_row'] = np.searchsorted(idx, D['start'].values.astype('datetime64[ns]'), side='left')
    D['e_row'] = np.searchsorted(idx, D['end'].values.astype('datetime64[ns]'), side='right') - 1
    D = D[(D['s_row'] >= 1) & (D['s_row'] < P.T) & (D['e_row'] >= D['s_row'])].sort_values(['code', 'start', 'ordr']).reset_index(drop=True)
    out = []
    for code, g in D.groupby('code', sort=False):
        g = g.reset_index(drop=True)
        i = 0
        while i < len(g):
            if not (g.loc[i, 'kind'] == 'regular' and (entry == 'all' or g.loc[i, 'nth'] == 1)):
                i += 1; continue
            r = g.loc[i]
            e = int(r['e_row']); n_ext = 0; jx = i + 1
            while jx < len(g) and g.loc[jx, 'announce'] <= P.IDX[e]:
                e = max(e, int(g.loc[jx, 'e_row'])); n_ext += 1; jx += 1
            out.append(dict(code=code, j=int(P.ci[code]), k=int(r['s_row']) - 1, s_row=int(r['s_row']), e_row=e, n_ext=n_ext, nth0=int(r['nth']),
                            announce0=r['announce'], start0=r['start'], end_final=P.IDX[e]))
            i = jx
    return pd.DataFrame(out).sort_values(['k', 'code']).reset_index(drop=True)


def make_signal_disp(P, univ, CH, shift=0, adv20=None, keep_info=True):
    """處置事件訊號（格式同 s4lib.make_signal，給 s10lib.simulate_s10 用）。
    位移 s：訊號列 k＋s（買在 k＋s＋1 開盤）、出場標記列 e_row＋s（賣在 e_row＋s＋1 開盤），整段往後移（同一批事件、同樣持有長度）。
    可買＝訊號列在 univ（只上市、流動性、非創新板）。同一天候選多於空位時，依訊號列「近 20 日平均成交值」由大到小（rank＝該值）。
    X 只在出場標記列為 True（該檔）；不在其他日子出場。持有中遇到同檔新事件不加碼（引擎本來就不重複買）。"""
    T, N = P.T, P.N
    sig = np.zeros((T, N), bool); X = np.zeros((T, N), bool); rank = np.full((T, N), np.nan)
    rows = {}
    n_all = n_ok = 0
    for r in CH.itertuples():
        k = r.k + shift; e = r.e_row + shift
        if k < 1 or k >= T - 1:
            continue
        n_all += 1
        if e < T:
            X[e, r.j] = True
        if not univ[k, r.j]:
            continue
        n_ok += 1
        sig[k, r.j] = True
        rank[k, r.j] = adv20[k, r.j] if adv20 is not None and np.isfinite(adv20[k, r.j]) else 0.0
        rows.setdefault(k, []).append(r.code)
    info = [dict(k=int(k), sig_date=str(P.IDX[k].date()), n_tgt=len(v), **({'targets': v} if keep_info else {})) for k, v in sorted(rows.items())]
    sigrows = X.any(axis=1)
    return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, sigrows=sigrows, RQ={}, n_events=n_all, n_eligible=n_ok)
