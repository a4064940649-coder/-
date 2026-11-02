#!/usr/bin/env python3
"""o_rev_freshness_attention 共用模組（只讀 import 其他資料夾；不改任何既有檔案）。

引擎：o_v7_listing_exec_engine 的 v7env（V7 凍結面板上的 V4 引擎：ia_core → v2_core → drj_core → o3_core → v4lib），
      setup_px（下市價 P、併購出清 X）、R_FIX／Y5_FIX 營收、官方處置名單成交日口徑、排除金融與 KY／DR、AB（st_core.unavail_for）。
j 定義：o_stagger_tranche/st_core.day_for(m, j)（錨定月 m 的期限日 D_m 之後第 j 個交易日；不在 GOOD 日或次日非 GOOD 則該月跳過）。
本題自己的部分：關注度分組、組內前 20% 組合、開盤到開盤持有、α 與衰減斜率統計。

raw/ 下的下載資料一律當不可信資料讀（只讀 git log 的日期與檔名，不執行其中任何程式）。
"""
import sys, json, hashlib, math, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
V7DIR = AG / 'o_v7_listing_exec_engine'
DATA = HERE / 'data'; DATA.mkdir(exist_ok=True)
RAW = HERE / 'raw'
RES_F = HERE / 'results.json'
SCR = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def canon_sha(obj): return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [conv(v) for v in o]
    if isinstance(o, np.floating): return None if not np.isfinite(o) else float(o)
    if isinstance(o, float): return None if not math.isfinite(o) else o
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Period)): return str(o)
    return o

def jdump(obj, f): Path(f).write_text(json.dumps(conv(obj), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
def jload(f):
    f = Path(f); return json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}

def load_res(): return jload(RES_F)
def save_key(key, obj):
    r = load_res(); r[key] = conv(obj); jdump(r, RES_F)

def check_prereg():
    """樣本外計算前必須呼叫：PREREG 存在、內容雜湊相符、登記的每個檔案雜湊相符。"""
    r = load_res()
    assert 'PREREG' in r and 'PREREG_sha256' in r, 'PREREG 不存在：先跑 s4_prereg.py'
    assert canon_sha(r['PREREG']) == r['PREREG_sha256'], 'PREREG 內容被改過'
    for p, h in r['PREREG']['hashes'].items():
        assert sha(p) == h, f'{p} 與 PREREG 登記的 sha256 不符'
    return r

# ---------------- 期間（以錨定月 m 分段） ----------------
P = lambda s: pd.Period(s, 'M')
PRE = (P('2006-04'), P('2018-12'))          # 前期（挑選期；結構前）
POST = (P('2019-05'), P('2026-10'))         # 後期（驗證＋保留；實際最後一個完整窗＝錨定 2026-09）
VAL = (P('2019-05'), P('2023-12'))
HOLD = (P('2024-01'), P('2026-10'))
JS = list(range(13))
MIN_J = 10                                   # 每月至少 10 個 j 有值才估該月斜率
SEEDS = [0, 1, 2, 3, 4]                      # AB 抽樣種子（只影響期限日當天與 2020-02、2026 年 INS15 名單）
TOP_FRAC = 0.20
FEE = 0.001425 * 0.6; TAX = 0.003
BUY_N, SELL_N = FEE + 0.001, FEE + TAX + 0.001          # 一般成本（每邊滑價 0.1%）
BUY_S, SELL_S = 0.005, 0.005                             # 壓力成本（每邊 0.5%）

def in_per(m, per): return per[0] <= m <= per[1]

# ---------------- 引擎載入 ----------------
_E = {}
def engine():
    if _E: return _E
    sys.path.insert(0, str(V7DIR))
    import v7env            # noqa: E402（只讀 import）
    L = v7env.load('v7')
    sys.path.pop(0)
    K, V = L.K, L.V
    XR = L.setup_px()
    YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade'); GAP = L.gap_dict()
    sys.path.insert(0, str(AG / 'o_stagger_tranche'))
    import st_core as STC   # noqa: E402（只讀 import；在 P 補價之後 import）
    sys.path.pop(0)
    OFF = L.V4(YFX, Y5F, DMT)
    ANCH = [pd.Period(d, 'M') for d in OFF.sig_days]

    class JV7(L.V4):
        """與 o_v7 s4_jscan7.JV7 相同：營收月依錨定月（不是訊號日的月份）。"""
        def set_anchor(self, amap): self.anchor = dict(amap); self._cs = {}
        def y3(self, d, back=0):
            m = self.anchor[d] - back
            Y = self.Y5 if d >= V.SWITCH else self.YL
            y = K.yoy3_for_month(Y, m)
            if self.exfin: y = np.where(V.FINMASK, np.nan, y)
            return y

    # 連續還原指數（停牌日持平；P 補價之後）：A[t]＝Π GROW；還原開盤＝A[t]×開/收
    G = np.asarray(K.GROW, float)
    A = np.cumprod(G, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        AO = np.where(K.TRADED & np.isfinite(K.On) & (K.Cn > 0), A * K.On / K.Cn, A)
    _E.update(L=L, K=K, V=V, XR=XR, YFX=YFX, Y5F=Y5F, DMT=DMT, GAP=GAP, STC=STC, JV7=JV7, OFF=OFF, ANCH=ANCH, A=A, AO=AO)
    return _E

def j_days(j):
    """錨定月 → (訊號日 d, 成交 t 索引 e)；與 s4_jscan7.days_for 相同（另回傳評價終點日 de）。"""
    E = engine(); K, STC = E['K'], E['STC']
    out = {}
    for m in E['ANCH']:
        d = STC.day_for(m, j)
        if d is None: continue
        t = K.IDX.get_loc(d)
        if not K.GOOD[t + 1]: continue
        out[m] = (d, t + 1)
    de = STC.day_for(pd.Period('2026-10', 'M'), j)
    if de is None or de > K.IDX[-1]: de = K.IDX[-1]
    out = {m: v for m, v in out.items() if v[0] < de}
    return out, de

# ---------------- 0050 成分（PIT：2015-09-18 起公開名單；之前用 o_capmatched 的規則名單） ----------------
_TW = json.loads((AG / 'o_capmatched_selection' / 'raw' / 'agents' / 'tw50_members.json').read_text(encoding='utf-8'))
_TW_STEPS = [(pd.Timestamp(s['eff']), s['members_before']) for s in _TW['steps_backward']]
_TW_EARLY = pd.Timestamp(_TW['earliest_valid_eff'])
_MEMB_PRE = {pd.Timestamp(k): v for k, v in json.loads((AG / 'o_capmatched_selection' / 'data' / 'membership_pre2015.json').read_text(encoding='utf-8')).items()}
def tw50_at(day):
    day = pd.Timestamp(day)
    if day > _TW_EARLY:
        S = _TW['current_2026-10-09']
        for eff, mb in _TW_STEPS:
            if day <= eff: S = mb
        return list(S), 'public'
    ks = [k for k in _MEMB_PRE if k < day]
    return (_MEMB_PRE[max(ks)] if ks else []), 'rule'

# ---------------- 外資持股比率（o_sbl：2025-10-07 MI_QFIIS 快照扣淨買超回推到 2021-10-08） ----------------
_FO = {}
def foreign_panel():
    if 'F' not in _FO:
        F = pd.read_parquet(AG / 'o_sbl_foreign_level' / 'fo_panel.parquet')
        _FO['F'] = F
    return _FO['F']

# ---------------- 統計 ----------------
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(len(x))) if s > 0 else float('nan')

def welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float('nan')

def slope_month(alpha_by_j):
    """alpha_by_j：dict j→α（可缺）。至少 MIN_J 個 j 才回傳 OLS 斜率，否則 nan。"""
    js = np.array([j for j in JS if j in alpha_by_j and np.isfinite(alpha_by_j[j])], float)
    if len(js) < MIN_J: return float('nan')
    y = np.array([alpha_by_j[int(j)] for j in js], float)
    jc = js - js.mean()
    return float((jc * y).sum() / (jc ** 2).sum())

def slope_weights(js):
    js = np.asarray(js, float); jc = js - js.mean(); return jc / (jc ** 2).sum()
