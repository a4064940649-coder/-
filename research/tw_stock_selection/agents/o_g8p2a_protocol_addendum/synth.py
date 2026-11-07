#!/usr/bin/env python3
"""合成前瞻帳本（只寫 scratchpad；欄位與 live_tracking/main_forward 相同）。探索的治理攻擊示範與判定器測試共用。"""
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd

FR_COLS = ['line', 'anchor', 'start', 'end', 'status', 'n_j', 'ret', 'ret_0050', 'excess_pp', 'ret_stress', 'excess_stress_pp', 'ret_cap10',
           'excess_cap10_pp', 'cost', 'n_hold', 'source', 'note']
TRADE_COLS = ['kind', 'line', 'date', 'code', 'side', 'is_odd', 'shares', 'fill_px', 'open_px', 'prev_close', 'close_px', 'ticks_from_open', 'sigma60_prev',
              'V60_prev', 'dec', 'oa_share_dec', 'OA_pred', 'share_of_open_vol', 'share_of_open_vol_actual', 'in_disposal', 'event_id', 'cost_rate',
              'share_of_qref', 'order_time', 'limit_px']


def make_ledger(d, months, rng=None, turnover=0.0, jitter=0.0, j0_void=True):
    """months：list of dict(anchor, x, void(bool)=False, open(bool)=False, nrows(int)=13)。
    每個 J 窗 excess_pp＝x＋雜訊、excess_stress_pp＝excess_pp − 0.5（較差者＝壓力線，判定線＝中位數＝x−0.5 附近）。
    void：7 個 J 窗不寫 meta（V0）→ 該月作廢。open：最後一個 J 窗 status＝open（或 nrows < 13）。
    回傳期望的逐月 (anchor, kind, x_med)，kind ∈ valid／void／open（只看帳本，不含 V3 規則）。"""
    rng = rng or np.random.default_rng(0)
    d = Path(d); shutil.rmtree(d, ignore_errors=True); (d / 'paper_lists').mkdir(parents=True)
    rows, trades, expect = [], [], []
    for m in months:
        an = m['anchor']; per = pd.Period(an, 'M')
        nrows = m.get('nrows', 13)
        void_lines = set(rng.choice(np.arange(1, 13), size=7, replace=False).tolist()) if m.get('void') else set()
        xs = []
        for j in range(nrows):
            st = pd.Timestamp(per.year, per.month, 10) + pd.Timedelta(days=j)
            en = st + pd.Timedelta(days=30)
            ex = m['x'] + (rng.normal(0, jitter) if jitter else 0.0)
            exs = ex - 0.5
            status = 'open' if (m.get('open') and j == nrows - 1) else 'complete'
            rows.append(dict(line=f'J{j}', anchor=an, start=str(st.date()), end=str(en.date()), status=status, n_j=1, ret=0.01, ret_0050=0.0,
                             excess_pp=ex, ret_stress=0.0, excess_stress_pp=exs, ret_cap10=0.0, excess_cap10_pp=ex, cost=0.0, n_hold=10,
                             source='synthetic', note=''))
            meta = dict(ts_ok=True, frozen_ts=f'{st.date()}T06:49:00+08:00', exec_open_ts=f'{(st + pd.Timedelta(days=1)).date()}T09:00:00+08:00',
                        rev_gate=dict(status='PASS', coverage=1.0))
            ok = True
            if j == 0 and j0_void: meta['rev_gate'] = dict(status='J0_UNVERIFIED'); ok = False
            if j in void_lines: ok = None
            if ok is not None:
                (d / 'paper_lists' / f'j{j}_{per.strftime("%Y%m")}.meta.json').write_text(json.dumps(meta), encoding='utf-8')
            if ok: xs.append(min(ex, exs))
            if turnover:
                trades.append(dict(kind='paper_whole', line=f'J{j}', date=str((st + pd.Timedelta(days=2)).date()), code='2330', side='buy', is_odd=0,
                                   shares=1000, fill_px=1e6 * turnover / 1000.0, open_px=1e6 * turnover / 1000.0))
        if m.get('open') or nrows < 13: expect.append((an, 'open', None))
        elif len(xs) >= 7: expect.append((an, 'valid', float(np.median(xs))))
        else: expect.append((an, 'void', None))
    pd.DataFrame(rows, columns=FR_COLS).to_csv(d / 'forward_returns.csv', index=False)
    pd.DataFrame(trades).reindex(columns=TRADE_COLS).to_csv(d / 'trades.csv', index=False)
    return expect


def months_seq(xs, start='2026-11', voids=(), opens=()):
    per = pd.Period(start, 'M'); out = []
    for i, x in enumerate(xs):
        out.append(dict(anchor=str(per + i), x=float(x), void=(i in voids), open=(i in opens)))
    return out
