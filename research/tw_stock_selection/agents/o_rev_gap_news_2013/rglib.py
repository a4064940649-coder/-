#!/usr/bin/env python3
"""o_rev_gap_news_2013 共用庫：只讀 import o_v4_revenue_patch_official/v4lib（V4 凍結版引擎），不改其他資料夾。
提供：V4 物件建構、缺口格清冊、PIT 代理 y3 的 callable、下市分類（併購／惡化）。
"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
V4DIR = AG / 'o_v4_revenue_patch_official'
sys.path.insert(0, str(V4DIR))
import numpy as np, pandas as pd
import v4lib as L          # noqa: E402
sys.path.pop(0)
K, V, R = L.K, L.V, L.R
IDX, COLS, N, CI = K.IDX, K.COLS, K.N, K.CI
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

W0, W1 = pd.Timestamp('2013-01-01'), pd.Timestamp('2021-12-31')     # 本題缺口窗（訊號日）

_C = {}
def base():
    """V4 凍結資料：R_FIX（YFX）、Y5_FIX（Y5F）、處置 trade 口徑、GAP、XR。"""
    if not _C:
        XR = L.setup_px()
        YFX, Y5F = L.load_rfix()
        _C.update(XR=XR, YFX=YFX, Y5F=Y5F, GAP=L.gap_dict(), DMT=L.disp_mask_fn('trade'))
    return _C

def mk(**kw):
    b = base()
    return L.V4(b['YFX'], b['Y5F'], b['DMT'], **kw)

# ---------------- 下市分類 ----------------
MC = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv', dtype=str)
MERGER = set(MC.code)
def gap_type(code, d):
    g = base()['GAP']
    if code in g and pd.Timestamp(d) <= pd.Timestamp(g[code]):
        return 'merger' if code in MERGER else 'deterior'
    return 'nondelist'
