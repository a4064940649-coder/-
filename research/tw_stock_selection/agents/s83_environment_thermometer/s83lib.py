#!/usr/bin/env python3
"""S83 環境溫度計：共用定義（事先寫死的 15 個環境指標、結果變數、統計工具）。
重用 S75／S76 的連續面板與引擎（全部不改）。一律 python3 -I -B。只診斷與監控，不是擇時規則、不改正式策略。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 's76_early_vs_late_diagnosis'))

TRAIN_END = '2015-12'      # 訓練段：結果視窗的最後一個月 ≤ 2015-12
VALID_START = '2016-01'    # 驗證段：指標月份 ≥ 2016-01
INDICATORS = [  # (代號, 白話名稱, 定義)；全部在「每個日曆月最後一個交易日 k 收盤後」可算，股池 U＝上市、非創新板、60 日均成交值≥1000 萬、股價≥10 元
    ('SPURT3', '飆股密度（3 個月）', 'U 內近 63 個交易日含息漲幅 > 50% 的檔數比例'),
    ('SPURT6', '飆股密度（6 個月）', 'U 內近 126 個交易日含息漲幅 > 50% 的檔數比例'),
    ('XSD21', '個股報酬離散度', 'U 內近 21 個交易日含息報酬的橫斷面標準差'),
    ('TAIL21', '右尾厚度', 'U 內近 21 日報酬的第 95 百分位 − 中位數'),
    ('BIGUP21', '大漲家數比例', '近 21 個交易日，每天 U 內單日漲幅 ≥ 6.5% 的檔數比例之平均（7%／10% 漲跌停年代都量得到）'),
    ('SMALLREL63', '中小型對大型相對強弱', 'U 內依 60 日均成交值分組：後 50% 等權近 63 日報酬 − 前 10% 等權近 63 日報酬'),
    ('TURN20_250', '市場成交值相對長期平均', '全部上市股成交值合計的 20 日平均 ÷ 250 日平均'),
    ('BREADTH60', '站上季線比例', 'U 內收盤在 60 日均線之上的檔數比例'),
    ('NSURGE3', 'SURGE≥3 的比例', '最新已公布營收月（訊號日 ≤ k）U 內 SURGE ≥ 3 的檔數 ÷ 有 SURGE 值的檔數（另報檔數）'),
    ('TOP10YOY', 'SURGE 前 10 名營收年增率中位數', '最新已公布營收月 U 內 SURGE 最高 10 檔的單月營收年增率中位數（營收月在 2013 年＝IFRS 斷點，缺值）'),
    ('REVBREADTH', '營收高成長家數比例', '最新已公布營收月 U 內單月營收年增率 > 20% 的檔數比例（2013 年缺值）'),
    ('WML3', '動能因子近 3 個月報酬', '每月底依「前 126 日到前 21 日」報酬把 U 分五等份，贏家等權 − 輸家等權的下一個月報酬；取最近 3 個已完成月份合計'),
    ('MKT12', '大盤 12 個月報酬', '0050 含息近 252 個交易日報酬'),
    ('MKTVOL63', '大盤波動', '0050 近 63 日日報酬標準差 × √252'),
    ('EX6', '策略自身近 6 個月超額', '現行帳戶近 6 個日曆月累積報酬 − 0050 同期（檢查超額有沒有延續性；不列入同步檢定）'),
]
IND = [x[0] for x in INDICATORS]
LEAD_OUT = ['lead1', 'lead3', 'lead6']       # 之後 1／3／6 個日曆月：帳戶累積報酬 − 0050 累積報酬
SYNC_OUT = ['sync1', 'sync3']                # 當月；近 3 個月（含當月）
SYNC_IND = [i for i in IND if i != 'EX6']    # EX6 與同步結果機械重疊，不檢定
SEC_OUT = ['sig_lead1', 'sig_lead3']         # 次要（不列入過關判定）：月底後第一個訊號期「SURGE 前 10 名（不看季線、無成本）− 0050」持有 1／3 期（S76 的 raw10）


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 12:
        return np.nan
    ra = pd.Series(a[m]).rank().to_numpy(); rb = pd.Series(b[m]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def tercile_means(x, y, lo, hi):
    """依訓練段門檻 lo／hi 分低（<lo）、中、高（>hi）三組，各組 y 的平均與筆數。"""
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    g = np.where(x < lo, 0, np.where(x > hi, 2, 1))
    out = {}
    for i, nm in enumerate(('low', 'mid', 'high')):
        s = m & (g == i)
        out[nm] = dict(n=int(s.sum()), mean=float(y[s].mean()) if s.any() else None)
    return out


def shift_null(X, Y, sign):
    """循環平移檢定（保留指標與結果各自的自我相關）：把指標整段循環平移 s 個月（s＝12…n−12），重算每一對的 sign×Spearman。
    X：{指標: 陣列}、Y：{結果: 陣列}、sign：{(指標, 結果): ±1}。回傳 obs{(i,o): 值}、個別 p、家族最大值校正 p（FWER）。"""
    keys = list(sign.keys()); n = len(next(iter(X.values())))
    obs = {k: sign[k] * spearman(X[k[0]], Y[k[1]]) for k in keys}
    shifts = list(range(12, n - 11))
    null = np.full((len(shifts), len(keys)), np.nan)
    for a, s in enumerate(shifts):
        for b, k in enumerate(keys):
            null[a, b] = sign[k] * spearman(np.roll(X[k[0]], s), Y[k[1]])
    mx = np.nanmax(null, axis=1)
    p_ind = {k: float((1 + np.sum(null[:, b] >= obs[k])) / (1 + len(shifts))) if np.isfinite(obs[k]) else None for b, k in enumerate(keys)}
    p_fwer = {k: float((1 + np.sum(mx >= obs[k])) / (1 + len(shifts))) if np.isfinite(obs[k]) else None for k in keys}
    return obs, p_ind, p_fwer, len(shifts)
