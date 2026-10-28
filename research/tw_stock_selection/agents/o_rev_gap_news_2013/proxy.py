#!/usr/bin/env python3
"""PIT 代理建構（步驟 2）：只用時間戳 ≤ 訊號日 13:30 的來源。
版本（事前列出）：
  N3：營收月 m−1、m−2、m−3 三個月年增率都有（官方值 ∪ PIT 新聞值）→ 平均；m−1 尚無任何值時，比照 AB 退一個月（m−2..m−4）
  N2：三個月中 ≥ 2 個月有值 → 有值月份平均
  N1：≥ 1 個月有值 → 有值月份平均
  Q ：季報法定期限滯後代理：法定期限 < 訊號日的最近一季（Q1 5/15、Q2 8/14（2013 年用 8/31，IFRS 首年從寬）、Q3 11/14、Q4＝年報 3/31），季營收年增率
  NQ：N1 有值用 N1，否則用 Q
新聞年增率：同一（代碼, 營收月）取訊號日前最早一則；文章有年增率直接用；沒有時以水準 ÷ 基期水準（官方 p−12 水準，或 PIT 新聞 p−12 水準）− 1。
驗證模式（only_news=True）：把三個月的官方值全部當成缺，只用新聞（＋季報）重建，模擬缺口格。
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd

AG = HERE.parent
CUT = pd.Timedelta(hours=13, minutes=30)
VERSIONS = ('N3', 'N2', 'N1', 'Q', 'NQ')

def q_deadline(qs):
    y, n = int(qs[:4]), int(qs[-1])
    return pd.Timestamp({1: f'{y}-05-15', 2: (f'{y}-08-31' if y == 2013 else f'{y}-08-14'), 3: f'{y}-11-14', 4: f'{y + 1}-03-31'}[n])

class Proxy:
    def __init__(self, mentions, LV, L5, Q, YL, Y5, switch):
        """mentions：parse_news 輸出；LV/L5：官方月營收水準（千元；長期 sii 優先／rev_5y）；Q：季營收（code,q,rev_k）；YL/Y5：V4 官方年增率面板。"""
        M = mentions.copy()
        M['ts'] = pd.to_datetime(M['ts'])
        M = M.sort_values('ts')
        self.M = {k: g[['ts', 'rev_k', 'yoy']].to_numpy() for k, g in M.groupby(['code', 'ym'])}
        self.LV, self.L5, self.YL, self.Y5, self.SW = LV, L5, YL, Y5, switch
        Qd = Q[~Q.suspect] if 'suspect' in Q.columns else Q
        self.Q = {(c, q): v for c, q, v in zip(Qd.code, Qd.q, Qd.rev_k)}
        self.qs = sorted({q for _, q in self.Q})
        allq = pd.period_range('2012Q1', '2021Q4', freq='Q')
        self.qdl = [(str(q).replace('Q', 'Q'), q_deadline(str(q))) for q in allq]

    # ---- 官方 ----
    def off_yoy(self, code, p, d):
        Y = self.Y5 if d >= self.SW else self.YL
        if code in Y.columns and p in Y.index:
            v = Y.at[p, code]
            if np.isfinite(v): return float(v)
        return np.nan
    def off_level(self, code, p):
        for X in (self.LV, self.L5):
            if code in X.columns and p in X.index:
                v = X.at[p, code]
                if np.isfinite(v) and v > 0: return float(v)
        return np.nan

    # ---- 新聞（PIT）----
    def news(self, code, p, d):
        """回傳 (yoy, level, ts)：訊號日 13:30 前最早一則。"""
        a = self.M.get((code, str(p)))
        if a is None: return np.nan, np.nan, None
        lim = pd.Timestamp(d) + CUT
        ok = a[a[:, 0] <= lim]
        if len(ok) == 0: return np.nan, np.nan, None
        lv = next((float(r[1]) for r in ok if np.isfinite(r[1])), np.nan)
        yy = next((float(r[2]) for r in ok if np.isfinite(r[2])), np.nan)
        return yy, lv, ok[0][0]
    def news_yoy(self, code, p, d, base_official=True):
        yy, lv, ts = self.news(code, p, d)
        if np.isfinite(yy): return yy
        if np.isfinite(lv):
            b = self.off_level(code, p - 12) if base_official else np.nan
            if not np.isfinite(b):
                _, b, _ = self.news(code, p - 12, d)
            if np.isfinite(b) and b > 0: return (lv / b - 1) * 100
        return np.nan

    def month_vals(self, code, d, back=0, only_news=False):
        m = pd.Period(d, 'M') - back
        out = []
        for p in (m - 1, m - 2, m - 3):
            v = np.nan if only_news else self.off_yoy(code, p, d)
            src = 'off'
            if not np.isfinite(v):
                v = self.news_yoy(code, p, d, base_official=True); src = 'news'
            out.append((v, src if np.isfinite(v) else None))
        return out

    def qproxy(self, code, d):
        d = pd.Timestamp(d)
        av = [q for q, dl in self.qdl if dl + pd.Timedelta(hours=23, minutes=59) <= d + CUT]
        if not av: return np.nan
        q = av[-1]; pq = pd.Period(q, 'Q')
        cur = self.Q.get((code, q))
        b = self.Q.get((code, str(pq - 4)))
        if b is None:   # 基期季：官方月營收三個月加總（當時已公告）
            ms = [pd.Period(pq.start_time, 'M') - 12 + k for k in range(3)]
            lv = [self.off_level(code, x) for x in ms]
            b = sum(lv) if all(np.isfinite(lv)) else None
        if cur is None or b is None or not b > 0: return np.nan
        return (cur / b - 1) * 100

    def y3(self, code, d, ver, only_news=False):
        """回傳 (proxy y3, 用到的新聞月數)。"""
        if ver == 'Q': return self.qproxy(code, d), 0
        mv = self.month_vals(code, d, 0, only_news)
        vals = [v for v, s in mv if s]
        nn = sum(1 for v, s in mv if s == 'news')
        if ver == 'N3':
            if len(vals) == 3 and nn >= (1 if not only_news else 3): return float(np.mean(vals)), nn
            if mv[0][1] is None:      # m−1 尚無值 → 比照 AB 退一個月
                mb = self.month_vals(code, d, 1, only_news)
                vb = [v for v, s in mb if s]; nb = sum(1 for v, s in mb if s == 'news')
                if len(vb) == 3 and nb >= (1 if not only_news else 3): return float(np.mean(vb)), nb
            return np.nan, 0
        need = {'N2': 2, 'N1': 1, 'NQ': 1}[ver]
        if len(vals) >= need and nn >= 1: return float(np.mean(vals)), nn
        if ver == 'NQ': return self.qproxy(code, d), 0
        return np.nan, 0

def load_inputs(K, mentions_file):
    d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str}); d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d['pri'] = (d['market'] != 'sii').astype(int); d = d.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
    LV = d.pivot(index='p', columns='code', values='revenue_k_twd')
    r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str}); r5['p'] = pd.PeriodIndex(r5['ym'], freq='M')
    L5 = r5.drop_duplicates(['p', 'code'], keep='last').pivot(index='p', columns='code', values='revenue')
    Q = pd.read_parquet(AG / 'o_v4_revenue_patch_official' / 'raw_quarterly.parquet')
    M = pd.read_parquet(mentions_file)
    return M, LV, L5, Q
