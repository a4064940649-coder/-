#!/usr/bin/env python3
"""o_rev_avail_ds_pit_count（o17 題 1）共用函式：只 import main_v2/v2_core（不改），重建主策略 C0 名單與股池、
訊號日分類（D<S／D=S／D>S）、營收缺值原因分類（ER2 定義延伸到 2013～2026，新增 POST13）、U_opt 放回排序。
本模組不計算任何報酬、淨值或 t 值。用法：被 s*.py import；一律 python3 -I -B。"""
import sys, csv, json, hashlib, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o17_revavail')
SCR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
COLS = list(K.COLS); CI = {c: j for j, c in enumerate(COLS)}; N = len(COLS)
IDX = K.IDX

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon_sha(o): return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
def jdump(o, f): Path(f).write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
def jload(f): return json.loads(Path(f).read_text(encoding='utf-8'))

ENGINE_SHA = {'v2_core': sha(AG / 'main_v2/v2_core.py'), 'ia_core': sha(AG / 'independent_audit/ia_core.py'),
              'tw_revenue': sha(K.LONG), 'rev_5y': sha(K.X / 'rev_5y.csv.gz')}

# ---------------- 期間 ----------------
SEG = {'S0613': ('2006-01-01', '2013-12-31'), 'S1418': ('2014-01-01', '2018-12-31'), 'S1324': ('2019-01-01', '2024-05-31'),
       'S2406': ('2024-06-01', '2026-12-31')}
def seg4(d):
    d = pd.Timestamp(d)
    for k, (a, b) in SEG.items():
        if pd.Timestamp(a) <= d <= pd.Timestamp(b): return k
    return None

# ---------------- 訊號日分類 ----------------
TCAL = V.TCAL
def deadline_engine(d):
    """v2_core.deadline_case 用的期限 D（訊號月 10 日起第一個 TCAL 交易日）；特例由 deadline_case 決定。"""
    return TCAL[TCAL >= pd.Timestamp(d.year, d.month, 10)][0]

SPECIAL_NOTE = {'2019-02': '2019-01 營收期限延到 2019-02-15（v2_core 判 D>S；st_core SPECIAL_DL）',
                '2023-04': '2023-03 營收期限延到 2023-04-12（v2_core 判 D>S；st_core SPECIAL_DL）',
                '2020-02': '2020-01 營收（疫情）可延到 2020-02-15；v2_core 硬判 D=S（實際法定期限晚於訊號日，等同 D>S）',
                '2013-02': '春節：10 日為假日，D＝S＝2013-02-18'}
SPECIAL_DL = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12'), '2020-02': pd.Timestamp('2020-02-17')}

def legal_deadline(d):
    m = str(pd.Period(d, 'M'))
    if m in SPECIAL_DL:
        x = SPECIAL_DL[m]; return TCAL[TCAL >= x][0]
    return deadline_engine(d)

# ---------------- C0 ----------------
YL, Y5, Y5F = V.load_revenue()

def build_c0(latest=False):
    """C0＝V2(gapfix, YL, Y5, exfin=True).plan(10, official)。回傳 (S, rows)。rows：每訊號日 dict。"""
    S = V.V2('gapfix', YL, Y5F if latest else Y5, latest=latest, exfin=True)
    pl = S.plan(10, 'official')
    rows = []
    for e, pk, el, d in pl:
        cs = S.cross_section(d)
        elm = cs['el']
        pr_rev = pd.Series(np.where(elm, cs['y3'], np.nan)).rank(pct=True).to_numpy()
        pr_52 = pd.Series(np.where(elm, cs['s52'], np.nan)).rank(pct=True).to_numpy()
        rows.append(dict(sig=d, exec=IDX[e], t=cs['t'], e=e, picks=[COLS[j] for j in pk],
                         pool=[COLS[j] for j in np.where(elm)[0]],
                         pr_rev={COLS[j]: float(pr_rev[j]) for j in np.where(elm)[0]},
                         pr_52={COLS[j]: float(pr_52[j]) for j in np.where(elm)[0]}))
    return S, rows

def c0_cached(latest=False):
    f = SCR / f'c0_{"latest" if latest else "formal"}.pkl'
    if f.exists():
        o = pickle.loads(f.read_bytes())
        if o.get('engine') == ENGINE_SHA: return o['rows']
    S, rows = build_c0(latest)
    f.write_bytes(pickle.dumps({'engine': ENGINE_SHA, 'rows': rows}))
    return rows

# ---------------- 營收水準（原因分類用） ----------------
_d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
_d['p'] = pd.PeriodIndex(pd.to_datetime(_d['month']), freq='M')
_d['pri'] = (_d['market'] != 'sii').astype(int)
_d = _d.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
LV = _d.pivot(index='p', columns='code', values='revenue_k_twd')
LV = LV.reindex(pd.period_range(LV.index.min(), LV.index.max(), freq='M'))
LASTREV = {c: LV[c].dropna().index.max() for c in LV.columns}
FIRSTREV = {c: LV[c].dropna().index.min() for c in LV.columns}
_r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str})
_r5['p'] = pd.PeriodIndex(_r5['ym'], freq='M')
_r5 = _r5.drop_duplicates(['p', 'code'], keep='last')
Y5COL = _r5.pivot(index='p', columns='code', values='yoy')
Y5COL = Y5COL.reindex(pd.period_range(Y5COL.index.min(), Y5COL.index.max(), freq='M'))
LAST5 = {c: Y5COL[c].dropna().index.max() for c in Y5COL.columns}
FIRST5 = {c: Y5COL[c].dropna().index.min() for c in Y5COL.columns}
NAME = {}
for c, nm in zip(_d['code'], _d['name']): NAME.setdefault(c, nm)
for r in csv.DictReader(open(K.X.parent / 'stock_list.csv', encoding='utf-8')): NAME.setdefault(r['code'], r['name'])
SNAPN = {}
for f in ('grs_568e6685_tools_twse_list.csv', 'grs_893c8772_grs_stock_no.csv', 'goristock_c3c718ab_grs_stock_no.csv'):
    p = AG / 'o_pit_shares_industry_2006_2013/raw/industry_snap' / f
    if p.exists():
        for r in csv.reader(open(p, encoding='utf-8', errors='replace')):
            r = [x.strip().strip('"') for x in r]
            if len(r) >= 2 and r[0].isdigit() and len(r[0]) == 4: SNAPN.setdefault(r[0], set()).add(r[1])
LMF = AG / 'o_v7_listing_exec_engine/listing_master.csv'
LM = {r['code']: r for r in csv.DictReader(open(LMF, encoding='utf-8'))}
DOF = AG / 'o_infra_v3_rebuild/delist_official.csv'
DOFF = {}
for r in csv.DictReader(open(DOF, encoding='utf-8')): DOFF.setdefault(r['code'], r['date'])
LT = {}
for j, c in enumerate(COLS):
    w = np.where(K.TRADED[:, j])[0]
    LT[c] = IDX[w[-1]] if len(w) else pd.NaT

def is_foreign(c):
    nms = {NAME.get(c, '') or ''} | SNAPN.get(c, set())
    return c[:2] == '91' or any(('KY' in n.upper()) or n.endswith('DR') or n.startswith('F-') for n in nms)

def delist_date(c):
    x = DOFF.get(c) or (LM.get(c) or {}).get('delist_date') or ''
    return pd.Timestamp(x) if x else pd.NaT

def month_cause_tw(c, x, d):
    """tw_revenue 期間（訊號日 < 2021-03-01）的單月缺值原因（ER2 erlib 定義＋POST13）。"""
    lr, fr = LASTREV[c], FIRSTREV[c]
    ltm = pd.Period(LT[c], 'M') if pd.notna(LT[c]) else None
    if x > lr:
        if ltm is not None and pd.Period('2009-12', 'M') <= ltm <= pd.Period('2012-12', 'M'): return 'TRUNC'
        if x >= pd.Period('2013-01', 'M') and lr >= pd.Period('2012-12', 'M'):
            return 'IFRS13' if pd.Timestamp(d) <= pd.Timestamp('2013-12-31') else 'POST13'
        return 'PRE12'
    if x < fr: return 'NEWLIST'
    return 'HOLE'

PRIO = ['TRUNC', 'IFRS13', 'POST13', 'POST13_R5', 'PRE12', 'HOLE', 'NEWLIST', 'BASE_LE0']
GOV = ('TRUNC', 'IFRS13', 'POST13', 'POST13_R5', 'PRE12', 'HOLE', 'NEVER_DOM')

def cause(c, d):
    """訊號日 d 的 y3 缺值原因。d < 2021-03-01 用 tw_revenue 水準（m−1..m−3、m−13..m−15）；之後用 rev_5y yoy 欄（m−1..m−3）。"""
    if is_foreign(c): return 'FOREIGN'
    m = pd.Period(d, 'M')
    if pd.Timestamp(d) < V.SWITCH:
        if c not in LV.columns: return 'NEVER_DOM'
        ks = set()
        for x in [m - k for k in (1, 2, 3)] + [m - k for k in (13, 14, 15)]:
            v = LV[c].get(x, np.nan)
            if not np.isfinite(v): ks.add(month_cause_tw(c, x, d))
            elif v <= 0 and x <= m - 13: ks.add('BASE_LE0')
    else:
        ks = set()
        if c not in Y5COL.columns:
            # rev_5y 完全沒有這檔：國內股且有 tw_revenue（2013 年後存活者缺口延伸）→ POST13_R5；否則 NEVER_DOM
            return 'POST13_R5' if c in LV.columns else 'NEVER_DOM'
        for x in [m - k for k in (1, 2, 3)]:
            v = Y5COL[c].get(x, np.nan)
            if not np.isfinite(v):
                if x > LAST5[c]: ks.add('POST13_R5')
                elif x < FIRST5[c]: ks.add('NEWLIST')
                else: ks.add('HOLE')
    for k in PRIO:
        if k in ks: return k
    return 'UNKNOWN'

# ---------------- U_opt（ER2 定義，延伸到任意期間） ----------------
class VU(V.V2):
    """mode：'base'（＝C0）、'opt'（受影響股營收百分位＝1.0）、'p90'、'carry'。lo/hi：放回期間（訊號日）。"""
    def __init__(self, mode='base', lo='2013-01-01', hi='2026-12-31', which='GOV', YLx=None, Y5x=None):
        self.mode, self.which = mode, which
        self.lo, self.hi = pd.Timestamp(lo), pd.Timestamp(hi)
        self.aff = {}; self.causes = {}
        super().__init__('gapfix', YL if YLx is None else YLx, Y5 if Y5x is None else Y5x, exfin=True)

    def cross_section(self, d):
        if d in self._cs: return self._cs[d]
        cs = super().cross_section(d)
        if self.mode == 'base' or not (self.lo <= d <= self.hi):
            self.aff[d] = []; return cs
        t, e = cs['t'], cs['e']
        v60 = self.V60[t]; px = K.Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True; top &= base
        s52 = self.S52[t]
        el0 = top & np.isfinite(s52) & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e] & ~V.FINMASK
        y3 = cs['y3'].copy()
        A = np.zeros(N, bool); cz = {}
        for j in np.where(el0 & ~np.isfinite(y3))[0]:
            k = cause(COLS[j], d); cz[COLS[j]] = k
            if self.which == 'ALL' or k in GOV: A[j] = True
        self.causes[d] = cz
        if self.mode == 'carry':
            for j in np.where(A)[0]:
                yy = np.nan
                for b in range(1, 25):
                    Y = self.Y5 if d >= V.SWITCH else self.YL
                    z = K.yoy3_for_month(Y, pd.Period(d, 'M') - b)[j]
                    if np.isfinite(z): yy = float(z); break
                y3[j] = yy
            A &= np.isfinite(y3)
        self.aff[d] = [COLS[j] for j in np.where(A)[0]]
        el = cs['el'] | A
        n = el.sum()
        s52p = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
        if self.mode == 'carry':
            rp = pd.Series(np.where(el, y3, np.nan)).rank(pct=True).to_numpy(); ytie = y3
        else:
            norm = el & ~A
            rp = pd.Series(np.where(norm, y3, np.nan)).rank().to_numpy() / n
            rp[A] = 1.0 if self.mode == 'opt' else 0.9
            ytie = np.where(A, (np.inf if self.mode == 'opt' else np.nan), y3)
            if self.mode == 'p90':
                ytie = np.where(A, np.nanpercentile(y3[norm], 90), y3)
        sc = 0.5 * s52p + 0.5 * rp
        sc[~el] = np.nan
        ytie = np.where(el, ytie, np.nan)
        cs2 = dict(cs, el=el, sc=sc, y3=ytie, A=A)
        self._cs[d] = cs2
        return cs2

def u_lists(S):
    pl = S.plan(10, 'official')
    out = {}
    for e, pk, el, d in pl:
        S.cross_section(d)
        out[str(d.date())] = dict(picks=[COLS[j] for j in pk], aff=list(S.aff.get(d, [])), causes=dict(S.causes.get(d, {})))
    return out
