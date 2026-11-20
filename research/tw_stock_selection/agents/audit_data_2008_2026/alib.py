#!/usr/bin/env python3
"""資料面稽核共用：自己讀原始資料、自己算參考價與含息日報酬（不 import 研究中心程式）。python3 -I -B"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
RS = Path('/home/user/-/research/tw_stock_selection')
S75 = RS / 'agents/s75_continuous_2008_2026'
TMP = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/audit75')
HERE = Path(__file__).resolve().parent
SEAM = pd.Timestamp('2015-01-05'); END = pd.Timestamp('2026-10-22')
FEE, TAX, MINFEE, XS = 0.001425 * 0.28, 0.003, 20.0, 0.003
ODD = pd.Timestamp('2020-10-26'); LIM10 = pd.Timestamp('2015-06-01')


def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0


def limup(ref, pct):
    tk = tick(ref * (1 + pct)); return np.floor(ref * (1 + pct) / tk + 1e-9) * tk


def limdn(ref, pct):
    tk = tick(ref * (1 - pct)); return np.ceil(ref * (1 - pct) / tk - 1e-9) * tk


def fee_of(sh, px):
    rnd = (sh // 1000) * 1000; odd = sh - rnd; f = 0.0
    if rnd > 0: f += max(MINFEE, rnd * px * FEE)
    if odd > 0: f += max(MINFEE, odd * px * FEE)
    return f


def trades():
    t = pd.read_csv(S75 / 'trades.csv.gz', dtype={'code': str}, parse_dates=['sig_date', 'ent_date', 'exit_date'])
    return t


def twt49u():
    fr = [pd.read_csv(f, dtype={'code': str}) for f in sorted(glob.glob(str(RS / 'raw/twt49u/twse-exright/*/*.csv.gz')))]
    e = pd.concat(fr, ignore_index=True); e['date'] = pd.to_datetime(e['date'])
    return e.drop_duplicates(['date', 'code'], keep='last')


class Data:
    """每檔一個日表（index＝連續日曆：Asoul 交易日 < 2015-01-05 ∪ 官方日 K 交易日 ≤ END）。
    欄：O、C、CHG（原始）、VOL、AMT、mk（1 上市 2 上櫃 0 不在檔內）、src（asoul／dpfill／off）、name。"""
    def __init__(self, codes=None):
        off = pd.read_parquet(TMP / 'off.parquet'); off = off[off['date'] <= END]
        a = pd.read_csv(RS / 'raw/pre2015/asoul_twse_2007_2015H1.csv.gz', dtype={'code': str}, parse_dates=['date'])
        self.asoul_all_dates = pd.DatetimeIndex(sorted(a['date'].unique()))
        pre = self.asoul_all_dates[self.asoul_all_dates < SEAM]
        offd = pd.DatetimeIndex(sorted(off['date'].unique()))
        self.IDX = pre.append(offd); self.npre = len(pre)
        self.off_last = off.groupby('code')['date'].max()
        self.asoul_last = a.groupby('code')['date'].max(); self.asoul_first = a.groupby('code')['date'].min()
        self.e50 = off[off['code'] == '0050'].set_index('date').sort_index()
        if codes is not None:
            off = off[off['code'].isin(codes)]; a = a[a['code'].isin(codes)]
        self.off = {c: g.set_index('date').sort_index() for c, g in off.groupby('code')}
        self.asoul = {c: g.set_index('date').sort_index() for c, g in a.groupby('code')}
        dp = RS / 'data_panel/panel'
        cols = None
        self.dp = {}
        for k in ('open', 'close_raw', 'ret', 'value', 'src', 'flag', 'adj', 'high'):
            x = pd.read_parquet(dp / f'{k}.parquet')
            if codes is not None:
                x = x[[c for c in x.columns if c in codes]]
            self.dp[k] = x
        self.ev = twt49u()
        self.evd = {c: g.set_index('date').sort_index() for c, g in self.ev.groupby('code')}
        self._cache = {}

    def frame(self, c):
        if c in self._cache:
            return self._cache[c]
        f = pd.DataFrame(index=self.IDX, columns=['O', 'C', 'CHG', 'VOL', 'AMT'], dtype=float)
        f['mk'] = 0; f['src'] = ''; f['name'] = ''; f['xflag'] = 0
        pre = self.IDX[:self.npre]
        if c in self.asoul:
            a = self.asoul[c].reindex(pre)
            m = a['close'].notna().to_numpy() | a['open'].notna().to_numpy() | a['volume'].notna().to_numpy()
            ii = pre[m]
            f.loc[ii, ['O', 'C', 'VOL', 'AMT']] = a.loc[ii, ['open', 'close', 'volume', 'amount']].to_numpy()
            ch = a.loc[ii, 'change'].to_numpy(float).copy(); cl = a.loc[ii, 'close'].to_numpy(float)
            ch[np.isnan(ch) & (cl > 0)] = 0.0                      # 空白漲跌價差且有成交 → 0（S75 規格）
            f.loc[ii, 'CHG'] = ch; f.loc[ii, 'mk'] = 1; f.loc[ii, 'src'] = 'asoul'; f.loc[ii, 'xflag'] = a.loc[ii, 'xflag'].fillna(0).astype(int).to_numpy()
        if c in self.dp['close_raw'].columns:                       # data_panel 補格
            dc = self.dp['close_raw'][c].reindex(pre); fill = dc.notna() & (f.loc[pre, 'src'] == '')
            ii = pre[fill.to_numpy()]
            if len(ii):
                r = self.dp['ret'][c].reindex(ii)
                f.loc[ii, 'C'] = dc.loc[ii]; f.loc[ii, 'O'] = self.dp['open'][c].reindex(ii); f.loc[ii, 'AMT'] = self.dp['value'][c].reindex(ii)
                chg = dc.loc[ii] - dc.loc[ii] / (1 + r)
                f.loc[ii, 'CHG'] = np.round(chg, 4); f.loc[ii, 'mk'] = 1; f.loc[ii, 'src'] = 'dpfill'
        if c in self.off:
            o = self.off[c]
            ii = o.index
            f.loc[ii, ['O', 'C', 'CHG', 'VOL', 'AMT']] = o[['open', 'close', 'change', 'volume', 'amount']].to_numpy()
            f.loc[ii, 'mk'] = np.where(o['market'] == 'twse', 1, 2); f.loc[ii, 'src'] = 'off'; f.loc[ii, 'name'] = o['name'].fillna('').to_numpy()
        # ---- 參考價與含息日報酬（自己寫；規則依 cslib 說明：REF＝收盤−漲跌價差；除權息日用 TWT49U 參考價）
        C = f['C'].to_numpy(float); CHG = f['CHG'].to_numpy(float).copy(); mk = f['mk'].to_numpy()
        has = np.isfinite(C) & (C > 0)
        CF = pd.Series(np.where(has, C, np.nan)).ffill().to_numpy()
        prevC = np.r_[np.nan, CF[:-1]]; prev_tr = np.r_[False, has[:-1]]
        zero = np.nan_to_num(CHG) == 0
        refx = np.full(len(C), np.nan); fixk = np.array([''] * len(C), dtype=object)
        if c in self.evd:
            e = self.evd[c]; e = e[e.index.isin(self.IDX)]
            refx[self.IDX.get_indexer(e.index)] = e['ref_price'].to_numpy(float)
        for i in np.flatnonzero(has & np.isfinite(refx) & (mk == 1) & zero):
            rr = C[i] / refx[i]
            if 0.899 <= rr <= 1.101:
                CHG[i] = C[i] - refx[i]; fixk[i] = 'twt49u'
            else:
                CHG[i] = C[i] - prevC[i]; fixk[i] = 'twt49u_oob_prevclose'
        for i in np.flatnonzero(has & np.isfinite(prevC) & zero & (np.abs(C - prevC) > 1e-9) & ~np.isfinite(refx) & ~prev_tr & (mk == 1)):
            if abs(C[i] / prevC[i] - 1) <= 0.105:
                CHG[i] = C[i] - prevC[i]; fixk[i] = 'prev_untraded'
        ref = C - CHG; ref[~(ref > 0)] = np.nan
        R = C / ref - 1; big = np.abs(R) > 0.5; R[big] = np.nan
        R = np.where(np.isfinite(R), R, np.where(has, C / prevC - 1, 0.0)); R = np.where(np.isfinite(R), R, 0.0)
        f['has'] = has; f['prevC'] = prevC; f['REF'] = np.where(np.isfinite(ref), ref, prevC); f['R'] = R; f['fix'] = fixk; f['bigR'] = big
        self._cache[c] = f
        return f
