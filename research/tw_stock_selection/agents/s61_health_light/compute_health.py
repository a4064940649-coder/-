#!/usr/bin/env python3
"""策略健康度燈號（S61 事前登記；SPEC.json／PREREG.json）——每日或每月都可以跑。

用途：管理預期、觸發資料檢查。**不用來擇時、不改交易規則、不改部位。**
三項（只用評估月月底以前、含當月可得的資料）：
  ① 滾動 12 個月已出場勝率 < 35%（出場日在近 12 個日曆月的已出場交易，單筆 proceeds÷cost−1 > 0 的比例）
  ② 滾動 12 個月單位部位超額 < 0（Σ 月超額 ÷ Σ 月平均個股部位比重 W_ACT，× 1200 換成年化個百分點；
     W_ACT > 0 時亮不亮只看 12 個月月超額加總的正負，W_ACT 估計只影響顯示數值）
  ③ 最近 3 個名單月選出檔數都 ≤ 2（名單月＝訊號日所在月；第一個到最後一個名單月之間缺的月份算 0 檔）
亮 0 項綠、1 項黃（顯示用）、2 項以上紅。

用法：
  python3 -I -B compute_health.py --nav 淨值.csv[.gz] --trades 交易.csv[.gz] --picks 名單.csv [--weights w_act.csv[.gz]]
         [--nav-col E0|W10_70] [--bench-col 0050] [--month YYYY-MM] [--history] [--out 輸出.json]
  淨值檔：date＋策略欄＋0050 欄（每日；例 xq_scripts/site/navs_official.csv.gz）。
  交易檔：code, ent_date, exit_date, cost, proceeds, why（why＝open_end 表示期末仍持有；例 official_trades.csv.gz）。
  名單檔：month（YYYY-MM）或 sig_date，加 code（一列一檔；code 空白＝該月 0 檔）或 n_picks（一列一月）（例 official_picks.csv）。
  W_ACT 檔（選填）：date, w_act（每日個股部位市值 ÷ 淨值）。沒給時用交易檔近似（每筆持有期間市值 cost→proceeds 依交易日幾何內插）。
輸出 JSON 到 stdout 或 --out。--history 另附逐月表。只用標準函式庫＋numpy／pandas。"""
import argparse, json, sys
import numpy as np
import pandas as pd

WIN_TH, UNIT_TH, PICK_TH = 0.35, 0.0, 2
WIN_N, PICK_N = 12, 3
DISCLAIMER = '健康度燈號只用來管理預期與觸發資料檢查，不是買賣訊號：不擇時、不減碼、不改交易規則。正式檢討仍看 S27 警戒線（A、E 與同月 2 條以上）。'


def lamp_table(M, first_signal=None):
    """M：以 PeriodIndex（月）為索引，欄位 excess（策略月報酬−0050 月報酬）、w_act、n_closed、n_win、n_picks（NaN＝該月沒有名單）。
    first_signal：該段第一個名單月（None＝取 n_picks 第一個非空月）。回傳逐月燈號表。"""
    M = M.sort_index().copy()
    fs = pd.Period(first_signal, 'M') if first_signal is not None else M['n_picks'].first_valid_index()
    after = M.index >= fs
    nc = M['n_closed'].where(after, 0).fillna(0); nw = M['n_win'].where(after, 0).fillna(0)
    ex = M['excess'].where(after); wa = M['w_act'].where(after)
    k = np.arange(len(M)); kfs = int(np.searchsorted(M.index.asi8, fs.ordinal))
    full12 = (k - kfs + 1) >= WIN_N
    full3 = (k - kfs + 1) >= PICK_N
    C = nc.rolling(WIN_N, min_periods=1).sum(); Wn = nw.rolling(WIN_N, min_periods=1).sum()
    win12 = (Wn / C).where(full12 & (C > 0))
    sx = ex.rolling(WIN_N, min_periods=WIN_N).sum(); sw = wa.rolling(WIN_N, min_periods=WIN_N).sum()
    unit12 = (sx / sw * 1200).where(full12 & (sw > 0))
    ex12 = (sx * 100).where(full12)
    pk = M['n_picks'].where(after)
    pk = pk.where(~after | pk.notna(), 0.0)                    # 第一個名單月之後缺名單＝0 檔（資料中斷也會被③抓到）
    last = M['n_picks'].last_valid_index()
    if last is not None:
        pk = pk.where(M.index <= last)                         # 最後一個名單月之後（本月名單還沒出）不補 0
    pmax3 = pk.rolling(PICK_N, min_periods=PICK_N).max().where(full3)
    out = pd.DataFrame(index=M.index)
    out['win12'] = win12; out['n_closed12'] = C.where(full12); out['unit12_pp'] = unit12; out['excess12_sum_pp'] = ex12
    out['picks_last3'] = [None] * len(M)
    for i in range(len(M)):
        if full3[i] and np.isfinite(pmax3.iloc[i]):
            out.iat[i, out.columns.get_loc('picks_last3')] = [int(v) for v in pk.iloc[i - 2:i + 1]]
    out['L1'] = win12 < WIN_TH
    out['L2'] = unit12 < UNIT_TH
    out['L3'] = pmax3 <= PICK_TH
    out['n_lit'] = out[['L1', 'L2', 'L3']].sum(axis=1).astype(int)
    out['valid'] = full12
    out['lamp'] = np.where(~full12, None, np.where(out['n_lit'] >= 2, '紅', np.where(out['n_lit'] == 1, '黃', '綠')))
    return out


def approx_weights(nav, trades):
    """沒有 W_ACT 檔時：每筆交易持有期（進場日～出場前一日；open_end 含最後一日）市值以 cost→proceeds 依交易日幾何內插，加總 ÷ 淨值。"""
    idx = nav.index; v = np.zeros(len(idx))
    for r in trades.itertuples(index=False):
        i0 = int(idx.searchsorted(pd.Timestamp(r.ent_date)))
        i1 = int(idx.searchsorted(pd.Timestamp(r.exit_date)))
        if r.why == 'open_end':
            i1 = min(i1 + 1, len(idx))
        n = i1 - i0
        if n <= 0 or not (r.cost > 0) or not (r.proceeds > 0):
            continue
        g = (r.proceeds / r.cost) ** (np.arange(1, n + 1) / max(n, 1))
        v[i0:i1] += r.cost * g
    return pd.Series(v / nav.to_numpy(), index=idx)


def monthly_from_files(nav_path, trades_path, picks_path, weights_path=None, nav_col='E0|W10_70', bench_col='0050'):
    N = pd.read_csv(nav_path, index_col=0, parse_dates=True).sort_index()
    nav, b = N[nav_col].astype(float), N[bench_col].astype(float)
    me = nav.groupby(nav.index.to_period('M')).tail(1).index
    pm = me.to_period('M')
    rs = nav.reindex(me).pct_change(); rb = b.reindex(me).pct_change()
    T = pd.read_csv(trades_path, dtype={'code': str})
    if weights_path:
        Wd = pd.read_csv(weights_path, index_col=0, parse_dates=True).iloc[:, 0]
        w_src = 'W_ACT 檔（精確）'
    else:
        Wd = approx_weights(nav, T)
        w_src = '由交易檔近似（只影響②顯示數值，不影響亮燈）'
    wm = Wd.groupby(Wd.index.to_period('M')).mean()
    X = T[T['why'] != 'open_end'].copy()
    X['ret'] = X['proceeds'] / X['cost'] - 1
    X['xm'] = pd.to_datetime(X['exit_date']).dt.to_period('M')
    nc = X.groupby('xm').size(); nw = X[X['ret'] > 0].groupby('xm').size()
    Pk = pd.read_csv(picks_path, dtype={'code': str})
    Pk['m'] = (pd.PeriodIndex(Pk['month'], freq='M') if 'month' in Pk else pd.to_datetime(Pk['sig_date']).dt.to_period('M'))
    if 'n_picks' in Pk:
        npk = Pk.groupby('m')['n_picks'].last().astype(float)
    else:
        npk = Pk.assign(h=Pk['code'].fillna('').str.strip().ne('')).groupby('m')['h'].sum().astype(float)
    # 名單月可能比淨值最後一個月還新（例：本月名單已出、淨值還沒更新）：只用到淨值最後一個月
    M = pd.DataFrame(index=pm)
    M['excess'] = (rs - rb).to_numpy(); M['ret_strategy'] = rs.to_numpy(); M['ret_0050'] = rb.to_numpy()
    M['w_act'] = wm.reindex(pm).to_numpy()
    M['n_closed'] = nc.reindex(pm, fill_value=0).to_numpy(); M['n_win'] = nw.reindex(pm, fill_value=0).to_numpy()
    M['n_picks'] = npk.reindex(pm).to_numpy()
    info = dict(nav_last_day=str(nav.index[-1].date()), w_source=w_src,
                picks_last_month=str(npk.index.max()), picks_newer_than_nav=[dict(month=str(m), n_picks=int(v)) for m, v in npk.items() if m > pm[-1]])
    return M, info


def explain(r, month):
    f = lambda x, d=1: '無資料' if x is None or not np.isfinite(x) else f'{x:.{d}f}'
    lamp = r['lamp']
    items = [
        dict(項目='① 滾動 12 個月已出場勝率', 數值=None if not np.isfinite(r['win12']) else round(float(r['win12']) * 100, 1), 單位='%',
             門檻='< 35% 亮', 亮=bool(r['L1']), 筆數=None if not np.isfinite(r['n_closed12']) else int(r['n_closed12']),
             說明=f"近 12 個月已出場 {f(r['n_closed12'], 0)} 筆，賺錢比例 {f(r['win12'] * 100 if np.isfinite(r['win12']) else np.nan)}%"),
        dict(項目='② 滾動 12 個月單位部位超額', 數值=None if not np.isfinite(r['unit12_pp']) else round(float(r['unit12_pp']), 1), 單位='個百分點／年（每 100% 個股部位）',
             門檻='< 0 亮', 亮=bool(r['L2']), 十二個月超額加總=None if not np.isfinite(r['excess12_sum_pp']) else round(float(r['excess12_sum_pp']), 1),
             說明=f"近 12 個月對 0050 月超額加總 {f(r['excess12_sum_pp'])} 個百分點，除以個股部位比重後年化 {f(r['unit12_pp'])} 個百分點"),
        dict(項目='③ 連續 3 個月選出檔數 ≤ 2', 數值=r['picks_last3'], 單位='檔（最近 3 個名單月）', 門檻='3 個月都 ≤ 2 亮', 亮=bool(r['L3']),
             說明=f"最近 3 個名單月選出 {r['picks_last3']} 檔"),
    ]
    txt = {'綠': '三項都正常。', '黃': '一項亮：留意，不代表策略失效（單項亮在策略有效時也常見）。',
           '紅': '兩項以上亮：狀態像 2011～2012 樣本外（選股命中率低）。請先檢查營收、名單、價格資料與管線是否正常；資料無誤時，代表未來 1～2 年落後 0050 的機率較高，期望應下修。不停用、不減碼。'}
    return dict(評估月=str(month), 燈號=lamp, 亮燈項數=int(r['n_lit']), 白話=txt.get(lamp, '資料不足，無法判斷。'), 三項=items)


def main(argv=None):
    ap = argparse.ArgumentParser(description='策略健康度燈號（S61）')
    ap.add_argument('--nav', required=True); ap.add_argument('--trades', required=True); ap.add_argument('--picks', required=True)
    ap.add_argument('--weights'); ap.add_argument('--nav-col', default='E0|W10_70'); ap.add_argument('--bench-col', default='0050')
    ap.add_argument('--month'); ap.add_argument('--history', action='store_true'); ap.add_argument('--out')
    a = ap.parse_args(argv)
    M, info = monthly_from_files(a.nav, a.trades, a.picks, a.weights, a.nav_col, a.bench_col)
    L = lamp_table(M)
    m = pd.Period(a.month, 'M') if a.month else L.index[-1]
    r = L.loc[m]
    res = dict(名稱='策略健康度燈號', 規則出處='research/tw_stock_selection/agents/s61_health_light（SPEC.json／PREREG.json，事前登記）',
               資料最後一天=info['nav_last_day'], 評估月是否完整=bool(m < pd.Period(info['nav_last_day'], 'M') or pd.Timestamp(info['nav_last_day']).is_month_end),
               W_ACT來源=info['w_source'], **explain(r, m), 免責=DISCLAIMER)
    if info['picks_newer_than_nav']:
        res['較新名單'] = dict(說明='名單比淨值新：③可先用最新名單看（①②要等淨值更新）', 名單=info['picks_newer_than_nav'])
        allpk = pd.concat([M['n_picks'].dropna(), pd.Series({pd.Period(x['month'], 'M'): x['n_picks'] for x in info['picks_newer_than_nav']})]).sort_index()
        last3 = [int(v) for v in allpk.iloc[-3:]]
        res['較新名單']['最近3個名單月檔數'] = last3
        res['較新名單']['③是否亮'] = bool(max(last3) <= PICK_TH)
    if a.history:
        H = L.copy(); H.index = H.index.astype(str)
        res['逐月'] = [dict(月=i, 燈號=x['lamp'], 勝率=None if not np.isfinite(x['win12']) else round(x['win12'], 4),
                          單位部位超額=None if not np.isfinite(x['unit12_pp']) else round(x['unit12_pp'], 2), 最近3月檔數=x['picks_last3'],
                          亮=[k for k in ('L1', 'L2', 'L3') if x[k]]) for i, x in H.iterrows()]
    s = json.dumps(res, ensure_ascii=False, indent=1, default=str)
    if a.out:
        open(a.out, 'w', encoding='utf-8').write(s)
    else:
        print(s)
    return res


if __name__ == '__main__':
    main()
