#!/usr/bin/env python3
"""台積電權重缺口與集中度因子的輸入（窗報酬）。
- 2330 在 0050 的權重：GitHub 搜尋 API 在本環境被封鎖、元大月報與公開說明書封存連不到 → 依研究計畫改用
  「shares_hist × close」市值近似：每個訊號日，上市股中市值前 50 名（近似 0050 成分）內，2330 市值占比。
  股數用 shares_hist 2026-10 快照（2025-09 以前無歷史股數，往回沿用；2330 股數 20 年幾乎不變）。
  用 ex_tsmc.csv（2025-09-26 起每日實際權重）與 etf0050_w.csv 校準：報告近似值與實際值的差。
- 集中度因子 CF＝市值前 10 名市值加權（訊號日權重、窗內買進持有，含息）− 主策略同池等權（AB_V4|EW）。
- r_ex＝去掉台積電的 0050：(r0050 − w·r2330)/(1 − w)。
價格來自共用面板（唯讀）；衍生窗序列存 data/tsmc_inputs.parquet（凍結雜湊）。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fam_common as C
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')

def main():
    adj = pd.read_parquet(P / 'adj.parquet'); cr = pd.read_parquet(P / 'close_raw.parquet')
    sh = pd.read_csv(X / 'shares_hist.csv', dtype={'code': str})
    sh = sh[sh.ym == sh.ym.max()].set_index('code')['shares']
    codes = [c for c in cr.columns if c in sh.index]
    g = C.GRID
    # 訊號日市值（asof）
    crg = cr[codes].reindex(cr.index.union(g)).ffill(limit=5).reindex(g)
    mcap = crg * sh.reindex(codes).to_numpy()[None, :]
    adjg = adj.reindex(adj.index.union(g)).ffill(limit=5).reindex(g)
    R = (adjg.shift(-1) / adjg - 1).iloc[:-1]           # 個股窗報酬（含息）
    rows = []
    for i, d in enumerate(g[:-1]):
        m = mcap.loc[d].dropna()
        m = m[m > 0]
        top50 = m.sort_values(ascending=False).iloc[:50]
        top10 = top50.iloc[:10]
        w2330 = float(top50.get('2330', np.nan) / top50.sum()) if '2330' in top50.index else np.nan
        r = R.loc[d]
        rt = r.reindex(top10.index)
        ok = rt.notna()
        r10 = float((rt[ok] * top10[ok]).sum() / top10[ok].sum()) if ok.any() else np.nan
        r50cw = r.reindex(top50.index); ok5 = r50cw.notna()
        rcw50 = float((r50cw[ok5] * top50[ok5]).sum() / top50[ok5].sum()) if ok5.any() else np.nan
        rows.append(dict(win=d, w2330_approx=w2330, r2330=float(r.get('2330', np.nan)), r_top10cw=r10, r_top50cw=rcw50,
                         n_top50=len(top50)))
    T = pd.DataFrame(rows).set_index('win')
    # 實際權重（2025-09-26 起）
    ex = pd.read_csv(X / 'ex_tsmc.csv', parse_dates=['date']).set_index('date')['w2330'] / 100
    exg = ex.reindex(ex.index.union(g)).ffill(limit=5).reindex(g[:-1])
    exg[g[:-1] < ex.index[0]] = np.nan
    T['w2330_actual'] = exg.to_numpy()
    B = pd.read_parquet(C.DATA / 'bench.parquet')
    T['r0050'] = B['r0050']; T['rEW'] = B['rEW_ABV4']
    T['CF'] = T['r_top10cw'] - T['rEW']
    # 校準：近似權重對實際權重（有重疊的窗）
    ov = T[['w2330_approx', 'w2330_actual']].dropna()
    T.attrs = {}
    cal = float((ov.w2330_actual - ov.w2330_approx).mean()) if len(ov) else np.nan
    T['w2330_cal'] = (T['w2330_approx'] + cal).clip(0, 0.99)
    T['w2330_used'] = T['w2330_actual'].fillna(T['w2330_cal'])
    w = T['w2330_used']
    T['r_ex'] = (T['r0050'] - w * T['r2330']) / (1 - w)
    T.to_parquet(C.DATA / 'tsmc_inputs.parquet')
    print('overlap windows', len(ov), 'mean actual-approx', cal, 'approx/actual corr', ov.corr().iloc[0, 1] if len(ov) > 2 else None)
    print(T.loc['2023-12-01':, ['w2330_approx', 'w2330_actual', 'w2330_used', 'r2330', 'r0050', 'r_ex', 'CF']].round(4).to_string())
    # 0050 近似檢查：top50 市值加權報酬 vs 0050 實際
    x = T[['r_top50cw', 'r0050']].dropna()
    print('top50cw vs 0050 corr', x.corr().iloc[0, 1], 'mean diff pp', (x.r_top50cw - x.r0050).mean() * 100)

if __name__ == '__main__':
    main()
