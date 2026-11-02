#!/usr/bin/env python3
"""前瞻紙上追蹤（G8）：用 tw-stock-screener 公開資料（history.csv.gz 上市＋上櫃官方日 K、exdiv_5y、rev_5y、rev_hist、disposal_5y、stock_list）
依凍結規格產生主版本 M（合併池前 400、n10、上櫃 ≤ 5 檔）、衛星 R2（只上櫃前 300、n10）與對照 BASE（上市前 300、n10）的名單，
並把已到期窗的開盤→開盤報酬記到 paper_nav.tsv。
用法：python3 -I -B s6_live_picks.py --data <tw-stock-screener>/data --signal YYYY-MM-DD [--out <資料夾>]
訊號日＝每月 10 日起第一個交易日（j0，與正式版相同）；收盤後執行，隔日開盤以限價（≤ 前收＋2 跳）買進，開盤漲停不買改遞補（名單已附備取 5 檔）。"""
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd

SPEC = {'M': dict(pool='ALL', K=400, n=10, cap=5), 'R2': dict(pool='TP', K=300, n=10, cap=None), 'BASE': dict(pool='TW', K=300, n=10, cap=None)}


def load(data):
    h = pd.read_csv(data / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    h = h[h.code.str.fullmatch(r'\d{4}')]
    C = h.pivot_table(index='date', columns='code', values='close'); O = h.pivot_table(index='date', columns='code', values='open')
    Vv = h.assign(v=h.close * h.volume).pivot_table(index='date', columns='code', values='v')
    ex = pd.read_csv(data / 'extras' / 'exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ref = C.shift(1).copy()
    for c, d, r in zip(ex.code, ex.date, ex.ref_price):
        if d in ref.index and c in ref.columns and np.isfinite(r): ref.at[d, c] = r
    R = (C / ref - 1).where(lambda x: x.abs() <= 0.105, 0.0).fillna(0.0)
    AI = (1 + R).cumprod().where(C.notna())
    return C, O, Vv, AI


def yoy3(data, sig):
    r = pd.read_csv(data / 'extras' / 'rev_5y.csv.gz', dtype={'code': str})
    try:
        r2 = pd.read_csv(data / 'extras' / 'rev_hist.csv.gz', dtype={'code': str}); r = pd.concat([r, r2]).drop_duplicates(['code', 'ym'], keep='last')
    except FileNotFoundError:
        pass
    m = pd.Period(sig, 'M') - (1 if sig.day >= 10 else 2)
    ms = [str(m - k) for k in range(3)]
    y = r[r.ym.isin(ms)].pivot_table(index='code', columns='ym', values='yoy')
    return y.reindex(columns=ms).mean(axis=1, skipna=False)


def picks(data, sig, C, O, Vv, AI):
    sl = pd.read_csv(data / 'stock_list.csv', dtype=str).set_index('code')
    fin = set(sl.index[sl.industry == '金融保險業']) | {c for c in C.columns if c[:2] == '28' or c in ('5876', '5880')}
    kydr = {c for c in C.columns if c[:2] == '91' or 'KY' in str(sl.name.get(c, '')).upper() or 'DR' in str(sl.name.get(c, '')).upper()}
    ds = pd.read_csv(data / 'extras' / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    t = C.index.get_loc(sig)
    nxt = C.index[t + 1] if t + 1 < len(C.index) else sig + pd.offsets.BDay(1)
    disp = set(ds[(ds.date <= sig) & (ds.start <= nxt) & (ds.end >= sig)].code)
    v60 = Vv.iloc[max(0, t - 59):t + 1].mean(); n60 = Vv.iloc[max(0, t - 59):t + 1].notna().sum()
    w = AI.iloc[max(0, t - 251):t + 1]; s52 = AI.iloc[t] / w.max(); n52 = w.notna().sum()
    y3 = yoy3(data, sig)
    mk = sl.market.reindex(C.columns)
    out = {}
    for nm, v in SPEC.items():
        base = (v60 >= 1e7) & (n60 >= 20) & (C.iloc[t] >= 10)
        if v['pool'] == 'TW': base &= mk == 'TWSE'
        elif v['pool'] == 'TP': base &= mk == 'TPEX'
        else: base &= mk.isin(['TWSE', 'TPEX'])
        pool = v60[base].sort_values(ascending=False).index[:v['K']]
        pool = [c for c in pool if c not in fin and c not in kydr and c not in disp and np.isfinite(s52.get(c, np.nan)) and n52.get(c, 0) >= 60 and np.isfinite(y3.get(c, np.nan))]
        df = pd.DataFrame({'s52': s52[pool], 'y3': y3.reindex(pool)})
        df['sc'] = (0.5 * df.s52.rank(pct=True) + 0.5 * df.y3.rank(pct=True)).round(10)
        df['mkt'] = mk.reindex(pool)
        df = df.reset_index().rename(columns={'index': 'code'}).sort_values(['sc', 'y3', 'code'], ascending=[False, False, True])
        sel = []; ntp = 0
        for r in df.itertuples():
            if v['cap'] is not None and r.mkt == 'TPEX' and ntp >= v['cap']: continue
            sel.append(r.code); ntp += r.mkt == 'TPEX'
            if len(sel) == v['n'] + 5: break
        out[nm] = dict(main=sel[:v['n']], reserve=sel[v['n']:], pool_size=len(pool))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--signal', required=True); ap.add_argument('--out', default=None)
    a = ap.parse_args(); data = Path(a.data); sig = pd.Timestamp(a.signal)
    C, O, Vv, AI = load(data)
    res = picks(data, sig, C, O, Vv, AI)
    print(json.dumps(res, ensure_ascii=False))
    if a.out:
        od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
        rows = [dict(signal=str(sig.date()), strat=k, rank=i + 1, code=c, role='main' if i < SPEC[k]['n'] else 'reserve') for k, v in res.items() for i, c in enumerate(v['main'] + v['reserve'])]
        f = od / 'targets.tsv'
        old = pd.read_csv(f, sep='\t', dtype=str) if f.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(rows).astype(str)]).drop_duplicates(['signal', 'strat', 'rank'], keep='last').to_csv(f, sep='\t', index=False)
