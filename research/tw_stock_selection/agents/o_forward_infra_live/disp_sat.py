#!/usr/bin/env python3
"""處置衛星前瞻帳本與紙上淨值（o_forward_infra_live）。

決策規則＝research/tw_stock_selection/disposal_today.py（2026-11-02 版，sha256 a95cf1d8…）：
  處置第一天 s 收盤買、最後一天 e 收盤賣；固定 10 槽；同檔重疊或首尾相接（start ≤ 上一筆採用的 end＋3 日曆日）只做第一筆；
  s 收盤漲幅 ≥ 9% 不買；e 收盤跌幅 ≤ −9% 或無價 → 順延到第一個非跌停日收盤賣；加權指數前一交易日收盤 < 200 日均（含該日）不開新倉；
  前一日 60 日均成交值 ≥ 1,000 萬、收盤 ≥ 10。
可執行層＝agents/o_disposal_official_fullhist frozen_spec 第四節 v1_cap20_shrink_defer（C1 進、CL 出）：
  Qref＝處置期內 V÷(270÷match_min)，處置期外（順延）＝25%×V；每筆 ≤ 20%×Qref（買縮單、賣順延）；
  單邊成本＝手續費 max(20 元, 0.1425%×0.28×成交值)＋max(一跳/收盤, 0.1%)＋0.5×σ60(t−1)×√(q/Qref)；賣出加證交稅 0.3%；
  壓力：每邊再加 0.4%、手續費 6 折。下單金額用事前值：min(槽金額, 20%×V60(t−1)×match_min/270)。
match_min：start ≥ 2026-08-10（新制 5 日處置、約 2 分鐘撮合）＝2；舊制第一次 5、第二次以上 20。
閒置資金放 0050 含息（買賣 0050 每邊手續費 0.1425%×0.28，賣出加 ETF 證交稅 0.1%）。
線：DISP_TWSE（上市，全部採用事件）、DISP_TWSE_FIRST（上市，只做第一次：排除 30 個交易日內的再處置）、DISP_TPEX（上櫃另列）。
時戳（A5）：每個事件第一次被帳本看到的時間（first_seen_ts，寫入後不改）必須 ≥ 公告時間（公告日 14:30 台北，保守最早值）
  而且 ≤ 進場日 s 13:30（台北）；不符合的事件標 LATE，不進紙上淨值。
"""
import math, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TW = HERE.parent.parent
FEE_N, FEE_S, TAX, ETF_TAX, BASE, STRESS_X = 0.001425 * 0.28, 0.001425 * 0.6, 0.003, 0.001, 0.001, 0.004
SLOTS, CAP, NEW_REGIME = 10, 0.20, pd.Timestamp('2026-08-10')
LEDGER_COLS = ['event_id', 'code', 'name', 'market', 'announce', 'start', 'end', 'regime', 'match_min', 'chain_take', 'second',
               'announce_ts', 'entry_deadline_ts', 'first_seen_ts', 'clock_kind', 'ts_ok']


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def load_frames(adir):
    """截斷資料夾 → 處置衛星需要的資料（上市＋上櫃）。"""
    import main_picks_today as MP
    adir = Path(adir)
    lst = pd.read_csv(adir / 'stock_list.csv', dtype=str)
    dsp = pd.read_csv(adir / 'extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    dsp = dsp[dsp['code'].map(MP.is4)].copy()
    h = pd.read_csv(adir / 'history.csv.gz', dtype={'code': str}, parse_dates=['date'])
    cal = pd.DatetimeIndex(sorted(h['date'].unique()))
    codes = set(dsp['code'])
    h = h[h['code'].isin(codes) & (h['volume'] > 0) & (h['close'] > 0)].drop_duplicates(['date', 'code'], keep='last')
    h['val'] = h['volume'] * h['close']
    piv = {k: h.pivot(index='date', columns='code', values=v).reindex(cal).astype(float) for k, v in [('C', 'close'), ('O', 'open'), ('V', 'val')]}
    ex = pd.read_csv(adir / 'extras/exdiv_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
    ex = ex[ex['code'].isin(codes) & (ex['ref_price'] > 0) & (ex['prev_close'] > 0)].copy()
    ex['f'] = ex['prev_close'] / ex['ref_price']; ex = ex[(ex['f'] >= 1.0) & (ex['f'] < 3.0)]
    R = MP.hist_returns(piv['C'], piv['O'], ex)
    V60p = piv['V'].rolling(60, min_periods=60).mean().shift(1)
    S60p = R.rolling(60, min_periods=40).std().shift(1)
    PC = piv['C'].ffill().shift(1)
    tx = pd.read_csv(adir / 'taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index()
    e50 = pd.read_csv(adir / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    return dict(lst=lst, dsp=dsp, cal=cal, C=piv['C'], O=piv['O'], V=piv['V'], R=R, V60p=V60p, S60p=S60p, PC=PC, tx=tx, e50=e50)


def events(fr):
    """處置記錄 → 事件表（含 s、e、chain_take、second、regime、match_min）。"""
    d = fr['dsp'].copy(); cal = fr['cal']; lst = fr['lst'].set_index('code')
    d['market'] = d['code'].map(lst['market']).fillna('?'); d['name'] = d['code'].map(lst['name']).fillna('')
    d = d.sort_values(['code', 'start', 'date', 'end']).reset_index(drop=True)
    si = cal.searchsorted(d['start'])                          # 第一個 ≥ start
    ei = cal.searchsorted(d['end'], side='right') - 1          # 最後一個 ≤ end
    d['s'] = [cal[i] if i < len(cal) else pd.NaT for i in si]
    d['e'] = [cal[i] if (i >= 0 and d['end'].iloc[k] <= cal[-1]) else pd.NaT for k, i in enumerate(ei)]
    d['si'] = si; d['ei'] = np.where(d['end'] <= cal[-1], ei, 10 ** 9)
    take, second = [], []
    for code, g in d.groupby('code', sort=False):
        last_end = None; prev = []
        for i in g.index:
            st = d.at[i, 'start']
            tk = not (last_end is not None and st <= last_end + pd.Timedelta(days=3))
            if tk: last_end = d.at[i, 'end']
            s_i = d.at[i, 'si']
            sec = any((pe >= s_i - 30) and (pe < s_i) for pe in prev)
            prev.append(d.at[i, 'ei'] if d.at[i, 'ei'] < 10 ** 9 else 10 ** 9)
            take.append((i, tk)); second.append((i, sec))
    d['chain_take'] = pd.Series(dict(take)).reindex(d.index).astype(bool)
    d['second'] = pd.Series(dict(second)).reindex(d.index).astype(bool)
    d['regime'] = np.where(d['start'] >= NEW_REGIME, 'new5', 'old')
    d['match_min'] = np.where(d['regime'] == 'new5', 2, np.where(d['second'].astype(bool), 20, 5))
    d['event_id'] = d['code'] + '_' + d['start'].dt.strftime('%Y%m%d') + '_' + d['end'].dt.strftime('%Y%m%d')
    d = d.rename(columns={'date': 'announce'})
    return d.sort_values(['announce', 'code', 'start']).reset_index(drop=True)


def update_ledger(ev, ledger_path, clock, clock_kind, asof):
    """把 asof 前已公告的事件寫進帳本；first_seen_ts 第一次寫入後不改。回傳帳本。"""
    tz = 'Asia/Taipei'
    old = pd.read_csv(ledger_path, dtype=str) if Path(ledger_path).exists() else pd.DataFrame(columns=LEDGER_COLS)
    seen = set(old['event_id'])
    rows = []
    for _, r in ev[ev['announce'] <= asof].iterrows():
        if r['event_id'] in seen: continue
        ann_ts = pd.Timestamp(r['announce'].date()).tz_localize(tz) + pd.Timedelta(hours=14, minutes=30)
        dl = pd.Timestamp(r['start'].date()).tz_localize(tz) + pd.Timedelta(hours=13, minutes=30)
        rows.append(dict(event_id=r['event_id'], code=r['code'], name=r['name'], market=r['market'], announce=str(r['announce'].date()),
                         start=str(r['start'].date()), end=str(r['end'].date()), regime=r['regime'], match_min=int(r['match_min']),
                         chain_take=int(bool(r['chain_take'])), second=int(bool(r['second'])), announce_ts=ann_ts.isoformat(),
                         entry_deadline_ts=dl.isoformat(), first_seen_ts=clock.isoformat(), clock_kind=clock_kind,
                         ts_ok=int(ann_ts <= clock <= dl)))
    L = pd.concat([old, pd.DataFrame(rows, columns=LEDGER_COLS).astype(str)], ignore_index=True) if rows else old
    L = L.sort_values(['announce', 'code', 'start']).reset_index(drop=True)
    from fwd_data import write_csv
    write_csv(L, ledger_path, gz=False)
    return L, len(rows)


def simulate(fr, ev, ledger, start, lines=('DISP_TWSE', 'DISP_TWSE_FIRST', 'DISP_TPEX'), stress=False, equity=1_000_000):
    """逐日紙上淨值（start 收盤＝equity 現金→全放 0050）；回傳 {line: (nav Series, fills DataFrame, decisions dict)}。"""
    cal = fr['cal']; days = cal[cal >= pd.Timestamp(start)]
    C, R, V, V60p, S60p, PC = fr['C'], fr['R'], fr['V'], fr['V60p'], fr['S60p'], fr['PC']
    e50 = fr['e50'].reindex(cal.union(fr['e50'].index)).ffill().reindex(cal)
    tx = fr['tx']
    okts = dict(zip(ledger['event_id'], ledger['ts_ok'].astype(int)))
    E = ev.set_index('event_id')
    fee = FEE_S if stress else FEE_N; X = STRESS_X if stress else 0.0
    out = {}
    for line in lines:
        mk = 'TPEX' if line == 'DISP_TPEX' else 'TWSE'
        cand = ev[(ev['market'] == mk) & ev['chain_take'].astype(bool)]
        if line == 'DISP_TWSE_FIRST': cand = cand[~cand['second'].astype(bool)]
        cand = cand[cand['s'].notna() & (cand['s'] >= days[0])] if len(days) else cand.iloc[:0]
        by_s = {s: g for s, g in cand.groupby('s')}
        idle = float(equity); pos = {}; nav = {}; fills = []; dec = {}
        if not len(days): out[line] = (pd.Series(dtype=float), pd.DataFrame(), dec); continue
        nav[days[0]] = idle
        for k, t in enumerate(days[1:], start=1):
            tp = days[k - 1]
            r50 = e50[t] / e50[tp] if np.isfinite(e50[t]) and np.isfinite(e50[tp]) else 1.0
            idle *= r50
            for p in pos.values():
                r = R.at[t, p['code']] if p['code'] in R.columns else np.nan
                if np.isfinite(C.at[t, p['code']]): p['val'] *= (1 + (r if np.isfinite(r) else 0.0))
            # 出場（先賣、空出槽位）
            for eid in list(pos):
                p = pos[eid]; c = p['code']
                if not (t >= p['e']): continue
                px, r = C.at[t, c], R.at[t, c]
                if not np.isfinite(px) or (np.isfinite(r) and r <= -0.09): continue           # 跌停或無價：順延
                inD = t <= p['e']
                qref = V.at[t, c] * (p['mm'] / 270.0 if inD else 0.25)
                sh_cap = math.floor(CAP * qref / px) if np.isfinite(qref) else 0
                n = min(p['sh'], max(sh_cap, 0))
                if n <= 0: continue
                sg = S60p.at[t, c]; q = n * px
                imp = max(float(tick(px)) / px, BASE) + (0.5 * sg * math.sqrt(q / qref) if np.isfinite(sg) and qref > 0 else 0.0) + X
                fpx = px * (1 - imp); v = n * fpx; fe = max(20.0, fee * v); tx_ = TAX * v
                frac = n / p['sh']
                idle += (v - fe - tx_) * (1 - fee)                                          # 款項回 0050（買 0050 手續費）
                fills.append(dict(date=str(t.date()), line=line, event_id=eid, code=c, side='sell', shares=n, close_px=px, fill_px=fpx,
                                  cost_rate=imp, fee=fe, tax=tx_, qref=qref, share_of_qref=q / qref if qref > 0 else np.nan, deferred=int(t > p['e'])))
                p['val'] *= (1 - frac); p['sh'] -= n
                if p['sh'] <= 0: del pos[eid]
            # 進場
            if t in by_s:
                txp = tx[tx.index < t]
                bear = bool(len(txp) >= 200 and txp.iloc[-1] < txp.iloc[-200:].mean())
                for eid, r_ in by_s[t].sort_values(['announce', 'code']).set_index('event_id').iterrows():
                    c = r_['code']
                    if okts.get(eid, 0) != 1: dec[eid] = 'LATE（決策時戳晚於進場日 13:30）'; continue
                    if bear: dec[eid] = 'SKIP 空頭開關'; continue
                    if any(p['code'] == c for p in pos.values()): dec[eid] = 'SKIP 同檔持有中'; continue
                    if len(pos) >= SLOTS: dec[eid] = 'SKIP 槽滿'; continue
                    px = C.at[t, c] if c in C.columns else np.nan; r = R.at[t, c] if c in R.columns else np.nan
                    if not np.isfinite(px): dec[eid] = 'SKIP 進場日無價'; continue
                    if np.isfinite(r) and r >= 0.09: dec[eid] = 'SKIP 收盤漲幅 ≥ 9%'; continue
                    v60, pc = V60p.at[t, c], PC.at[t, c]
                    if not (np.isfinite(v60) and v60 >= 1e7 and np.isfinite(pc) and pc >= 10): dec[eid] = 'SKIP 流動性'; continue
                    navp = idle + sum(p['val'] for p in pos.values())
                    mm = int(r_['match_min'])
                    amt = min(navp / SLOTS, CAP * v60 * mm / 270.0)
                    qref = V.at[t, c] * mm / 270.0
                    amt = min(amt, CAP * qref) if np.isfinite(qref) else 0.0                      # 實際單盤量檢查（縮單）
                    sg = S60p.at[t, c]
                    imp = max(float(tick(px)) / px, BASE) + (0.5 * sg * math.sqrt(amt / qref) if np.isfinite(sg) and qref > 0 else 0.0) + X
                    fpx = px * (1 + imp); n = math.floor(amt / fpx)
                    if n <= 0: dec[eid] = 'SKIP 可成交股數 0'; continue
                    v = n * fpx; fe = max(20.0, fee * v); need = (v + fe) / (1 - fee - ETF_TAX)      # 賣 0050 籌款
                    if need > idle: dec[eid] = 'SKIP 現金不足'; continue
                    idle -= need
                    pos[eid] = dict(code=c, sh=n, val=n * px, e=E.at[eid, 'e'] if pd.notna(E.at[eid, 'e']) else pd.Timestamp('2100-01-01'), mm=mm)
                    dec[eid] = 'BUY'
                    fills.append(dict(date=str(t.date()), line=line, event_id=eid, code=c, side='buy', shares=n, close_px=px, fill_px=fpx,
                                      cost_rate=imp, fee=fe, tax=0.0, qref=qref, share_of_qref=v / qref if qref > 0 else np.nan, deferred=0))
            # 未到期而 e 已確定的持股：更新 e
            for eid, p in pos.items():
                if p['e'].year == 2100 and pd.notna(E.at[eid, 'e']): p['e'] = E.at[eid, 'e']
            nav[t] = idle + sum(p['val'] for p in pos.values())
        out[line] = (pd.Series(nav), pd.DataFrame(fills), dec)
    return out
