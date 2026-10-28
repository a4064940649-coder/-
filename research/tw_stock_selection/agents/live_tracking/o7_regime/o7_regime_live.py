#!/usr/bin/env python3
"""o7 前瞻登記：權值集中狀態輪動（凍結版，來源 agents/o_concentration_regime/，PREREG 見該資料夾 results.json）。

每月換股日（每月第一個日期 ≥ 11 的交易日）收盤後執行：
  python3 -I -B agents/live_tracking/o7_regime/o7_regime_live.py --data /home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data
輸出／追加：agents/live_tracking/o7_regime/ledger.csv（每個決策日一列；已存在的列不覆寫）。

凍結規則（不得修改；修改等於新研究，需要新的 PREREG）：
  股池：上市普通股（4 碼、非 0 開頭、排除 KY 與 91xx），決策日 60 日均成交值（min 50 筆）≥ 1,000 萬、收盤 ≥ 10 元，取前 300 名。
  top5＝前 5 大 60 日均成交值總和 ÷ 池內 300 檔總和。
  S2＝top5(本次決策日) − top5(6 個決策日之前)。S2 > 0 → 集中期 C：0050 100%；否則擴散期 D：主策略 100%。（h＝0，無遲滯帶）
  執行：決策日隔一個交易日收盤移轉（主策略在該日開盤照常換股，之後整腿賣出或買回）。
  對照：同平均權重靜態混合（0050 權重＝挑選期平均 w̄，見 registration.json），每月同日調回。
紙上淨值：主策略腿月報酬需由主研究員的主策略前瞻追蹤提供（main_leg_returns.csv：start,end,ret）；沒有時該列淨值留空，只記狀態。
序貫判定：見 registration.json 的 sequential_rule。"""
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
REG = json.loads((HERE / 'registration.json').read_text(encoding='utf-8'))


def load_hist(data):
    h = pd.read_csv(Path(data) / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    sl = pd.read_csv(Path(data) / 'stock_list.csv', dtype=str)
    tw = set(sl[sl.market == 'TWSE'].code)
    ky = {c for c, n in zip(sl.code, sl.name) if isinstance(n, str) and 'KY' in n.upper()}
    h = h[h.code.isin(tw) & (h.code.str.len() == 4) & ~h.code.str.startswith('0') & ~h.code.str.startswith('91') & ~h.code.isin(ky)]
    h['value'] = h.volume * h.close
    V = h.pivot_table(index='date', columns='code', values='value').sort_index()
    C = h.pivot_table(index='date', columns='code', values='close').sort_index()
    return V, C


def decision_dates(ix):
    s = pd.Series(ix, index=ix)
    return [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([ix.year, ix.month]) if (g.dt.day >= 11).any()]


def top5_share(V, C, d):
    v60 = V.loc[:d].tail(60)
    n = v60.notna().sum(); m = v60.mean()
    m = m[(n >= 50) & (m >= 1e7) & (C.loc[d] >= 10)]
    m = m.sort_values(ascending=False).head(300)
    return float(m.head(5).sum() / m.sum()), int(len(m)), list(m.head(5).index)


def msprt(x, sigma, tau):
    n = len(x)
    if n == 0: return float('nan')
    xb = float(np.mean(x)); s2, t2 = sigma ** 2, tau ** 2
    return math.sqrt(s2 / (s2 + n * t2)) * math.exp(n * n * t2 * xb * xb / (2 * s2 * (s2 + n * t2)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--e50', default='/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')
    a = ap.parse_args()
    V, C = load_hist(a.data)
    dd = decision_dates(V.index)
    led_p = HERE / 'ledger.csv'
    led = pd.read_csv(led_p, parse_dates=['decision']) if led_p.exists() else pd.DataFrame(columns=['decision'])
    have = set(led['decision']) if len(led) else set()
    rows = []
    shares = {}
    for d in dd:
        try:
            shares[d] = top5_share(V, C, d)
        except Exception:
            continue
    keys = sorted(shares)
    for i, d in enumerate(keys):
        if d < pd.Timestamp(REG['live_start']) or d in have: continue
        if i < 6: continue
        t5, npool, names = shares[d]; t5_6 = shares[keys[i - 6]][0]
        S2 = t5 - t5_6; state = 'C' if S2 > 0 else 'D'
        rows.append(dict(decision=d, top5=t5, top5_6ago=t5_6, S2=S2, state=state, w0050=1.0 if state == 'C' else 0.0, w0050_static=REG['wbar_frozen'],
                         n_pool=npool, top5_names=' '.join(names), recorded_utc=pd.Timestamp.utcnow().isoformat()))
    if rows:
        led = pd.concat([led, pd.DataFrame(rows)], ignore_index=True) if len(led) else pd.DataFrame(rows)
        led.to_csv(led_p, index=False)
    # 紙上淨值（需要主策略腿月報酬）
    mp = HERE / 'main_leg_returns.csv'
    if mp.exists() and len(led) >= 2:
        ml = pd.read_csv(mp, parse_dates=['start', 'end'])
        e = pd.read_csv(a.e50, parse_dates=['date']).set_index('date')['adj'].sort_index()
        recs = []; wprev = None
        c = REG['costs_normal']   # (主腿買, 主腿賣, 0050 買, 0050 賣)
        for _, r in led.iterrows():
            m = ml[ml.start == r['decision']]
            if not len(m): continue
            m = m.iloc[0]
            r50 = float(e.asof(m['end']) / e.asof(m['start']) - 1); rm = float(m['ret'])
            w = float(r['w0050']); wb = float(r['w0050_static'])
            sw = 0.0 if wprev is None else (abs(w - wprev) * ((c[1] + c[2]) if w > wprev else (c[3] + c[0])))
            recs.append(dict(start=m['start'], rot=w * r50 + (1 - w) * rm - sw, static=wb * r50 + (1 - wb) * rm, r0050=r50, main=rm, switch_cost=sw))
            wprev = w
        if recs:
            R = pd.DataFrame(recs); R['diff_vs_static'] = R.rot - R.static; R['diff_vs_0050'] = R.rot - R.r0050
            sq = REG['sequential_rule']['fallback']
            R['mSPRT_Lambda'] = [msprt(R.diff_vs_static.iloc[:k + 1].to_numpy(), sq['sigma'], sq['tau']) for k in range(len(R))]
            R.to_csv(HERE / 'paper_nav.csv', index=False)
    print(led.tail(3).to_string())


if __name__ == '__main__':
    main()
