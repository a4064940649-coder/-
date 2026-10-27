#!/usr/bin/env python3
"""上櫃下市股：覆蓋檢查＋合併版（tpex_main COMB_top400 n30）2019-06～2022-12 的期望拖累界限。PREREG 之後執行。
python3 -I -B tpex_scan.py → tpex_scan.json
來源：raw/henry753951_finance/data/stocks（FinMind TaiwanStockPrice，2013-01～2022-12-30；工作樹由 git checkout 取出，未執行任何程式）。
(1) 找出最後一筆落在 2019-05-17～2022-11-30（早於來源終點）、且不是轉上市的代碼 → 上櫃下市候選；對照 tpex_main 上櫃面板的最後一筆。
(2) 面板缺（或截斷）的候選股：每月月底判定是否「可能入池」（60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10），
    入池者以 30/400 的機率入選、取代 1/30 權重 → 每月 Δ＝Σ(r_s − r_合併版)/400；下市當月最後一筆之後：上界 0%（以最後收盤出清）、下界 −50%（經營惡化）。
(3) 把 Δ 加到 tpex_main 的合併版淨值，報 2021-01～2022-12 與 2019-06～2022-12 年化變動。"""
import json
from pathlib import Path
import pandas as pd, numpy as np
HERE = Path(__file__).resolve().parent; HE = HERE / 'raw' / 'henry753951_finance' / 'data' / 'stocks'
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); assert RES.get('PREREG')
SP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
TP = pd.read_parquet(SP / 'tpex_panel' / 'close_raw.parquet'); TW = pd.read_parquet(SP / 'panel' / 'close_raw.parquet')
tpl = TP.apply(lambda s: s.last_valid_index()); twf = TW.apply(lambda s: s.first_valid_index())
NAV = pd.read_parquet(HERE.parent / 'tpex_main' / 'navs_n30.parquet')
TWSE_DL = set(pd.read_csv(HERE / 'delist_official.csv', dtype=str)['code'])   # 證交所終止上市名單（上市股，不是上櫃）
rows = []; series = {}
files = sorted(HE.glob('*.csv'))
for f in files:
    c = f.stem
    if not (len(c) == 4 and c.isdigit()): continue
    try: d = pd.read_csv(f, usecols=['date', 'close', 'spread', 'Trading_money', 'Trading_Volume'])
    except Exception: continue
    d = d[d['Trading_Volume'] > 0]
    if d.empty: continue
    d['date'] = pd.to_datetime(d['date'])
    last = d['date'].iloc[-1]
    if not (pd.Timestamp('2019-05-17') <= last <= pd.Timestamp('2022-11-30')): continue
    transfer = c in TW.columns and pd.notna(twf[c]) and twf[c] > last - pd.Timedelta(days=30)
    in_twse_end = c in TW.columns and pd.notna(TW[c].loc[:last].last_valid_index()) and abs((TW[c].loc[:last].last_valid_index() - last).days) <= 3
    otc = not transfer and not in_twse_end and c not in TWSE_DL
    tl = tpl[c] if c in TP.columns else None
    rows.append({'code': c, 'henry_last': str(last.date()), 'otc': otc, 'in_tpex_panel': c in TP.columns, 'tpex_last': str(tl.date()) if tl is not None and pd.notna(tl) else None,
                 'tpex_gap_days': (last - tl).days if tl is not None and pd.notna(tl) else None})
    if otc:
        ref = d['close'] - d['spread']
        r = np.where(ref > 0, d['close'] / ref - 1, 0.0)
        series[c] = pd.DataFrame({'ret': r, 'close': d['close'].to_numpy(), 'val': d['Trading_money'].to_numpy()}, index=d['date'])
R = pd.DataFrame(rows)
miss = R[R['otc'] & (R['tpex_gap_days'].fillna(9999) > 5)]
out = {'n_codes_ending_2019-05-17_2022-11': int(len(R)), 'n_otc_delisted_candidates': int(R['otc'].sum()),
       'n_otc_full_in_tpex_panel': int((R['otc'] & (R['tpex_gap_days'].fillna(9999) <= 5)).sum()),
       'n_otc_missing_or_truncated': int(len(miss)), 'missing': miss.to_dict(orient='records')}
# (2)(3) 期望拖累
for col in ('base|COMB_top400|n30|slip0.001', 'exFIN|COMB_top400|n30|slip0.001'):
    nav = NAV[col].dropna()
    me = nav.groupby(nav.index.to_period('M')).last(); mr = me.pct_change()
    res = {}
    for tag, tail in (('upper_lastclose', 0.0), ('lower_minus50', -0.5)):
        delta = pd.Series(0.0, index=mr.index); nin = pd.Series(0, index=mr.index)
        for c in miss['code']:
            s = series[c]
            # 面板缺的日子才算（tpex 面板最後一筆之後）
            tl = tpl[c] if c in TP.columns and pd.notna(tpl.get(c)) else pd.Timestamp('1900-01-01')
            s = s[s.index > tl]
            if s.empty: continue
            v60 = s['val'].rolling(60, min_periods=40).mean()
            g = (1 + s['ret']).groupby(s.index.to_period('M')).prod() - 1
            lastm = s.index[-1].to_period('M')
            for m in g.index:
                prev = s[s.index.to_period('M') < m]
                if prev.empty: continue
                ok = (v60.loc[prev.index[-1]] >= 1e7 if np.isfinite(v60.loc[prev.index[-1]]) else False) and prev['close'].iloc[-1] >= 10
                if not ok or m not in mr.index or not np.isfinite(mr.loc[m]): continue
                rs = g.loc[m] if m != lastm else (1 + g.loc[m]) * (1 + tail) - 1
                delta.loc[m] += (rs - mr.loc[m]) / 400.0; nin.loc[m] += 1
        adj = (1 + mr.fillna(0) + delta).cumprod()
        def cagr(x, a, b):
            x = x[(x.index >= pd.Period(a, 'M')) & (x.index <= pd.Period(b, 'M'))]
            return float((1 + x).prod() ** (12 / len(x)) - 1) * 100 if len(x) else np.nan
        res[tag] = {w: {'orig': cagr(mr.fillna(0), a, b), 'adj': cagr(mr.fillna(0) + delta, a, b), 'diff_pp': cagr(mr.fillna(0) + delta, a, b) - cagr(mr.fillna(0), a, b)}
                    for w, (a, b) in {'2019-06_2022-12': ('2019-06', '2022-12'), '2021-01_2022-12': ('2021-01', '2022-12'), '2021-01_2026-10': ('2021-01', '2026-10')}.items()}
        res[tag]['stock_months_in_pool'] = int(nin.sum())
    out[col] = res
(HERE / 'tpex_scan.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print({k: v for k, v in out.items() if k != 'missing'})
print(miss.head(40).to_string())
