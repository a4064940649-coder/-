#!/usr/bin/env python3
"""o_rev_event_clock_ts 前瞻紀錄器：營收事件時鐘（EC）對官方日主策略（OFF）的配對紀錄（規則凍結於 results.json PREREG_FORWARD）。

建議執行時點：每日流程第 0 步（o_forward_infra_live/daily_step0.sh）之後，台北 06:49～08:50（成交日開盤前），每天一次；
可在晚上 22:40 後再跑一次（只會多記首見、不會重複決策）。
  python3 -I -B research/tw_stock_selection/agents/o_rev_event_clock_ts/forward/event_clock_log.py
預設資料（只讀）：o_forward_infra_live/raw/screener/data（第 0 步每天 fetch 的上游稀疏 clone）、live_tracking/main_forward（官方日名單）。
本程式只寫自己的 --out（預設本資料夾），絕不修改 o_forward_infra_live 與 live_tracking/main_forward。

凍結規則（V2；選版只看周轉，未看任何報酬）：
- 錨定月 m：營收月 m−1 的公告窗＝上一個官方成交日之後到本月官方成交日之前。官方訊號日 s_m＝m 月第一個 ≥ 11 日的交易日，官方成交日＝s_m 的下一個交易日。
- 每次執行：成交日 e＝執行時鐘之後第一個交易日 09:00 開盤。事件＝本次第一次看到的 (代碼, m−1) 營收（首見＝本機執行時鐘；同時記上游 rev 檔最後 commit 時間與 MOPS ann_date）。
- 分數：資料最後一個交易日 t 收盤；已首見 m−1 營收者用 m−1..m−3 平均年增率，其餘沿用 m−2..m−4；0.5×52 週比百分位＋0.5×營收百分位；
  股池＝main_picks_today.score_on（前 300、收盤 ≥ 10、排除金融、非處置；處置＝disposal_5y ∪ warnings_hist 處置）。
- 只有事件股可以觸發交易：(1) 持有中的事件股名次 > 20 或不在股池 → 賣；(2) 未持有的事件股名次 ≤ 10，依名次：持股 < 10 檔直接買；
  否則與「本月尚未交易過、分數最低的持股」比較，分數差 ≥ 0.05 才換；(3) 仍有空位 → 用全池名次 ≤ 10 的最佳未持有股補。每檔每月最多買 1 次、賣 1 次。
- 官方成交日：EC 校正成官方名單（OFF 同日換成官方名單）。官方名單優先讀 live_tracking/main_forward/paper_lists/official_YYYYMM.csv（in_top10＝1），
  再來是 corrections/official_YYYYMM_corrected.csv，都沒有才用 main_picks_today 自算（並註記）。
- 成交：隔日開盤。買進遇開盤漲停（未持有）→ 不成交、留空位等下次補；賣出遇開盤跌停或停牌 → 不成交、續抱到官方日校正。
- 成本（紙上）：一般＝手續費 0.1425%×0.28＋K5T 衝擊 max(0.1%＋0.5·σ60·√(Q/OA_pred), 一跳/開盤)，賣方加稅 0.3%；壓力＝手續費 ×0.6、每邊再加 0.4%。
  Q＝級距/10（100 萬為主欄，另記 300 萬、1,000 萬的整張／零股股數與衝擊估計）；OA_pred＝成交值十分位開盤占比×V60（V7 model_spec7）。
- 漏跑：成交日開盤前沒有執行到的交易日記入 ec_missing.csv，不回補；之後才首見的事件照常在當次成交，並記錄延遲。
輸出（--out）：ec_firstseen.csv、ec_decisions.csv、ec_state.json、ec_nav.csv、ec_monthly.csv、ec_missing.csv、ec_runs.jsonl。
"""
import argparse, datetime, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
AG = TOP.parent
TW = AG.parent
TZ = 'Asia/Taipei'
K_OUT, DELTA, N_HOLD, ENTRY_RANK = 20, 0.05, 10, 10
TIERS = [1_000_000, 3_000_000, 10_000_000]
FEE_N, FEE_S, TAX, EXTRA_S = 0.001425 * 0.28, 0.001425 * 0.6, 0.003, 0.004
START_ANCHOR = '2026-11'

sys.path.insert(0, str(TW))
import main_picks_today as MP   # noqa: E402（只讀 import）
sys.path.pop(0)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tick(p): return float(np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0))

SPEC = json.loads((AG / 'o_v7_listing_exec_engine' / 'data' / 'model_spec7.json').read_text())
OASH = np.array([SPEC['oa_share_by_dec'][str(i)] for i in range(1, 11)])

DEC_COLS = ['run_clock', 'portfolio', 'anchor', 'asof', 'exec_date', 'code', 'side', 'reason', 'rank', 'score', 'yoy3', 's52', 'is_event',
            'first_seen_local', 'upstream_rev_commit', 'ann_date', 'lag_td', 'status', 'open', 'prev_close', 'limit_up', 'limit_dn', 'open_limit',
            'close', 'in_disposal', 'lots_traded', 'odd_shares_traded', 'sigma60', 'v60', 'dec', 'oa_pred',
            'k5t_100w', 'k5t_300w', 'k5t_1000w', 'lots_100w', 'odd_100w', 'lots_300w', 'odd_300w', 'lots_1000w', 'odd_1000w']


class Cal:
    def __init__(self, D, hcal):
        h = json.loads((D / 'extras/tw_holidays.json').read_text(encoding='utf-8'))['years']
        self.hol = set(sum(h.values(), [])); self.years = set(h.keys()); self.hcal = set(pd.DatetimeIndex(hcal).normalize())
    def is_td(self, d):
        d = pd.Timestamp(d).normalize()
        if d in self.hcal: return True
        return d.weekday() < 5 and str(d.date()) not in self.hol
    def next_td(self, d, incl=False):
        d = pd.Timestamp(d).normalize()
        if not incl: d += pd.Timedelta(days=1)
        while not self.is_td(d): d += pd.Timedelta(days=1)
        return d
    def prev_td(self, d):
        d = pd.Timestamp(d).normalize() - pd.Timedelta(days=1)
        while not self.is_td(d): d -= pd.Timedelta(days=1)
        return d
    def signal_day(self, m):
        return self.next_td(pd.Timestamp(m.year, m.month, 11), incl=True)
    def tds(self, a, b):
        """(a, b] 的交易日。"""
        out = []; d = pd.Timestamp(a).normalize()
        while True:
            d = self.next_td(d)
            if d > pd.Timestamp(b).normalize(): return out
            out.append(d)


def git1(repo, *a):
    try:
        return subprocess.run(['git', '-C', str(repo), *a], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        return ''


def load_disposal(D, asof):
    d5 = pd.read_csv(D / 'extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    wp = D / 'extras/warnings_hist.csv.gz'
    if wp.exists():
        w = pd.read_csv(wp, dtype={'code': str}, parse_dates=['date', 'start', 'end'])
        w = w[w['flag'].astype(str).str.contains('處置')]
        d5 = pd.concat([d5[['date', 'code', 'start', 'end']], w[['date', 'code', 'start', 'end']]], ignore_index=True)
    d5 = d5[d5['date'] <= asof].sort_values('date').drop_duplicates(['code', 'start', 'end'], keep='first')
    return d5


def official_list(m, OFFD, data, Y3, dsp, cal):
    ym = f'{m.year:04d}{m.month:02d}'
    p = OFFD / 'paper_lists' / f'official_{ym}.csv'
    if p.exists():
        x = pd.read_csv(p, dtype={'code': str})
        if 'in_top10' in x: x = x[x['in_top10'].astype(str).isin(['1', 'True', 'true'])]
        x = x.sort_values('rank') if 'rank' in x else x
        return list(x['code'])[:N_HOLD], f'paper_lists/{p.name}'
    p = OFFD / 'corrections' / f'official_{ym}_corrected.csv'
    if p.exists():
        x = pd.read_csv(p, dtype={'code': str})
        x = x[x['in_top10'].astype(str).isin(['1', 'True', 'true'])].sort_values('rank')
        return list(x['code'])[:N_HOLD], f'corrections/{p.name}'
    s = cal.signal_day(m)
    if s not in data['cal']: return None, 'missing'
    order, _, _, _ = MP.score_on(data, Y3, dsp, s, ex_fin=True)
    return list(order.index[:N_HOLD]), 'self_computed(main_picks_today.score_on ex_fin, 處置聯集)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(AG / 'o_forward_infra_live' / 'raw' / 'screener' / 'data'))
    ap.add_argument('--official', default=str(AG / 'live_tracking' / 'main_forward'))
    ap.add_argument('--e50', default=str(AG / 'o_forward_infra_live' / 'raw' / 'finlab' / 'etf_0050.csv'))
    ap.add_argument('--out', default=str(HERE))
    ap.add_argument('--sim-clock', help='只供測試：模擬時鐘（必須搭配非預設 --out）')
    a = ap.parse_args()
    D = Path(a.data); OFFD = Path(a.official); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    test = a.sim_clock is not None
    if test:
        assert OUT.resolve() != HERE.resolve(), '--sim-clock 只能寫到測試資料夾'
        clock = pd.Timestamp(a.sim_clock).tz_localize(TZ) if pd.Timestamp(a.sim_clock).tzinfo is None else pd.Timestamp(a.sim_clock).tz_convert(TZ)
    else:
        clock = pd.Timestamp.now(tz=TZ)
        res = json.loads((TOP / 'results.json').read_text(encoding='utf-8'))
        pf = res.get('PREREG_FORWARD'); assert pf, 'PREREG_FORWARD 不存在：凍結前不得寫入正式前瞻紀錄'
        assert pf['file_sha256']['forward/event_clock_log.py'] == sha(__file__), 'event_clock_log.py 與 PREREG_FORWARD 登記的 sha256 不符'
    run = dict(clock=clock.isoformat(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), script_sha256=sha(__file__),
               mp_sha256=sha(TW / 'main_picks_today.py'), data_head=git1(D.parent, 'log', '-1', '--format=%H %cI'),
               rev_commit=git1(D.parent, 'log', '-1', '--format=%cI', '--', 'data/extras/rev_hist.csv.gz'))
    # ---------- 資料 ----------
    data = MP.load_prices(D, pd.Timestamp('2100-01-01'))
    hcal = data['cal']; t_last = hcal[-1]
    cal = Cal(D, hcal)
    cnow = clock.tz_localize(None).normalize()
    e = cnow if (cal.is_td(cnow) and clock.hour < 9) else cal.next_td(cnow)
    need = cal.prev_td(e)
    run.update(asof=str(t_last.date()), exec_date=str(e.date()))
    Y, Y3, _ = MP.load_revenue(D)
    dsp = load_disposal(D, t_last)
    rh = pd.read_csv(D / 'extras/rev_hist.csv.gz', dtype=str)
    ann = {(c, y): d for c, y, d in zip(rh.code, rh.ym, rh.ann_date) if isinstance(d, str) and d}
    r5 = pd.read_csv(D / 'extras/rev_5y.csv.gz', dtype={'code': str})
    # ---------- 首見 ----------
    fs_f = OUT / 'ec_firstseen.csv'
    fs = pd.read_csv(fs_f, dtype=str) if fs_f.exists() else pd.DataFrame(columns=['code', 'ym', 'first_seen_local', 'upstream_rev_commit', 'ann_date', 'bootstrap'])
    boot = len(fs) == 0
    seen = set(zip(fs.code, fs.ym))
    st_f = OUT / 'ec_state.json'
    S = json.loads(st_f.read_text(encoding='utf-8')) if st_f.exists() else None
    m_e = pd.Period(e, 'M')
    lo_ym = str(m_e - 3)
    pairs = {(c, y) for c, y in zip(r5.code, r5.ym) if y >= lo_ym} | {(c, y) for c, y in zip(rh.code, rh.ym) if y >= lo_ym}
    add = [(c, y, clock.isoformat(), run['rev_commit'], ann.get((c, y), ''), '1' if boot else '0') for c, y in sorted(pairs - seen)]
    if add:
        fs = pd.concat([fs, pd.DataFrame(add, columns=fs.columns)], ignore_index=True); fs.to_csv(fs_f, index=False)
    fsd = {(c, y): (fl, uc, an, b) for c, y, fl, uc, an, b in zip(fs.code, fs.ym, fs.first_seen_local, fs.upstream_rev_commit, fs.ann_date, fs.bootstrap)}
    # ---------- 狀態 ----------
    dec_f = OUT / 'ec_decisions.csv'
    DEC = pd.read_csv(dec_f, dtype={'code': str}) if dec_f.exists() else pd.DataFrame(columns=DEC_COLS)
    if S is None:
        p0 = OFFD / 'corrections' / 'official_202610_corrected.csv'
        x = pd.read_csv(p0, dtype={'code': str}); x = x[x['in_top10'].astype(str) == '1'].sort_values('rank')
        H0 = list(x['code'])[:N_HOLD]
        S = dict(created=clock.isoformat(), init_list=H0, init_source=str(p0.relative_to(AG)), init_exec='2026-10-13',
                 ec=H0, off=H0, last_clock=None, last_exec=None, anchor=None, traded_in=[], traded_out=[], processed=[], reconciled=[])
    # ---------- 漏跑紀錄 ----------
    miss_f = OUT / 'ec_missing.csv'
    miss = pd.read_csv(miss_f, dtype=str) if miss_f.exists() else pd.DataFrame(columns=['exec_date', 'noted_at', 'reason'])
    if S['last_exec']:
        gap = [d for d in cal.tds(S['last_exec'], e) if d < e]
        new_m = [(str(d.date()), clock.isoformat(), '成交日開盤前沒有執行紀錄器（不回補）') for d in gap if str(d.date()) not in set(miss.exec_date)]
        if new_m: miss = pd.concat([miss, pd.DataFrame(new_m, columns=miss.columns)], ignore_index=True)
    else:
        first = pd.Timestamp('2026-11-02')
        new_m = [(str(d.date()), clock.isoformat(), '紀錄器啟動前（2026-11 公告窗已開始）') for d in cal.tds(first - pd.Timedelta(days=1), e) if d < e]
        if new_m: miss = pd.concat([miss, pd.DataFrame(new_m, columns=miss.columns)], ignore_index=True)
    miss.to_csv(miss_f, index=False)
    stale = t_last < need
    run['stale'] = bool(stale)
    # ---------- 先補已成交決策的成交資料 ----------
    DEC = fill_exec(DEC, data, dsp, cal, D, S)
    # ---------- 決策 ----------
    rows = []
    s_m = cal.signal_day(m_e); x_m = cal.next_td(s_m)
    anchor = m_e if e <= x_m else m_e + 1
    if str(anchor) < START_ANCHOR: anchor = pd.Period(START_ANCHOR, 'M')
    if S['anchor'] != str(anchor):
        S.update(anchor=str(anchor), traded_in=[], traded_out=[], processed=[])
    rm = str(anchor - 1)
    already = (S['last_exec'] == str(e.date()))
    x_a = cal.next_td(cal.signal_day(anchor))
    if stale:
        # 不在這裡記缺漏：同一成交日開盤前若有較晚的執行拿到新資料，仍可決策；沒有的話下一次執行的缺口檢查會記入 ec_missing.csv
        run['decision'] = f'資料過期（最後 {t_last.date()} < 應有 {need.date()}）：本次不做決策；{e.date()} 開盤前請在上游更新後重跑'
    elif already:
        run['decision'] = '本成交日已決策過，只更新首見與成交資料'
    elif e == x_a:
        O, src = official_list(anchor, OFFD, data, Y3, dsp, cal)
        run['official_source'] = src
        if O is None:
            run['decision'] = '官方名單不存在、也無法自算：官方日校正延後（記缺漏）'
        else:
            for c in [h for h in S['ec'] if h not in O]: rows.append(dict(portfolio='EC', code=c, side='sell', reason='official_reconcile'))
            for c in [j for j in O if j not in S['ec']]: rows.append(dict(portfolio='EC', code=c, side='buy', reason='official_reconcile'))
            for c in [h for h in S['off'] if h not in O]: rows.append(dict(portfolio='OFF', code=c, side='sell', reason='official'))
            for c in [j for j in O if j not in S['off']]: rows.append(dict(portfolio='OFF', code=c, side='buy', reason='official'))
            S['ec'] = list(O); S['off'] = list(O); S['reconciled'].append(str(anchor))
            run['decision'] = f'官方日校正（{src}）'
    elif e < x_a and e >= pd.Timestamp(anchor.year, anchor.month, 1):
        # 分數（t_last 收盤；已首見 rm 者用新值）
        rowu = pd.Period(t_last, 'M') - 1
        Pn, Po = pd.Period(rm, 'M'), pd.Period(rm, 'M') - 1
        cols = Y3.columns
        new = Y3.loc[Pn].reindex(cols) if Pn in Y3.index else pd.Series(np.nan, index=cols)
        old = Y3.loc[Po].reindex(cols) if Po in Y3.index else pd.Series(np.nan, index=cols)
        known = pd.Series([(c, rm) in fsd for c in cols], index=cols)
        Ym = Y3.copy()
        if rowu not in Ym.index: Ym.loc[rowu] = np.nan
        Ym.loc[rowu] = new.where(known & new.notna(), old).reindex(Ym.columns)
        order, df, _, nel = MP.score_on(data, Ym, dsp, t_last, ex_fin=True)
        rank = pd.Series(np.arange(1, len(order) + 1), index=order.index)
        rk = lambda c: int(rank[c]) if c in rank.index else 10 ** 6
        scr = lambda c: float(df['score'].get(c, np.nan)) if c in df.index else float('nan')
        proc = set(S['processed'])
        ev = sorted({c for (c, y) in fsd if y == rm and c not in proc and c in df.index})
        S['processed'] = sorted(proc | set(ev))
        H = list(S['ec']); tin, tout = set(S['traded_in']), set(S['traded_out'])
        for h in list(H):
            if h in ev and h not in tin and h not in tout and (rk(h) > K_OUT or not bool(df['el'].get(h, False))):
                H.remove(h); tout.add(h); rows.append(dict(portfolio='EC', code=h, side='sell', reason='event_exit', is_event=1))
        for c in sorted([j for j in ev if j not in H and rk(j) <= ENTRY_RANK and j not in tout], key=rk):
            if len(H) < N_HOLD:
                H.append(c); tin.add(c); rows.append(dict(portfolio='EC', code=c, side='buy', reason='event_fill', is_event=1)); continue
            el_ = [h for h in H if h not in tin and h not in tout]
            if not el_: break
            w = min(el_, key=lambda h: (np.nan_to_num(scr(h), nan=-9), h))
            if np.nan_to_num(scr(c), nan=-9) - np.nan_to_num(scr(w), nan=-9) >= DELTA:
                H.remove(w); tout.add(w); H.append(c); tin.add(c)
                rows.append(dict(portfolio='EC', code=w, side='sell', reason=f'event_swap_out_for_{c}', is_event=0))
                rows.append(dict(portfolio='EC', code=c, side='buy', reason=f'event_swap_in_for_{w}', is_event=1))
        if ev:
            for c in order.index:
                if len(H) >= N_HOLD: break
                if c not in H and c not in tout and rk(c) <= ENTRY_RANK:
                    H.append(c); tin.add(c); rows.append(dict(portfolio='EC', code=c, side='buy', reason='fill_any', is_event=int(c in ev)))
        S['ec'] = H; S['traded_in'] = sorted(tin); S['traded_out'] = sorted(tout)
        for r in rows:
            c = r['code']; f = fsd.get((c, rm))
            r.update(rank=rk(c), score=scr(c), yoy3=float(df['yoy3'].get(c, np.nan)), s52=float(df['s52'].get(c, np.nan)))
            if f:
                lag = None
                if f[2]:
                    lag = len([d for d in cal.tds(pd.Timestamp(f[2]), e)]) - 1
                r.update(first_seen_local=f[0], upstream_rev_commit=f[1], ann_date=f[2], lag_td=lag)
        run['decision'] = f'公告窗：事件 {len(ev)} 檔，EC 決策 {len(rows)} 筆'
        run['n_events'] = len(ev); run['n_el'] = int(nel)
    else:
        run['decision'] = '公告窗外（官方日之後、下月公告前）：不交易'
    for r in rows:
        r.update(run_clock=clock.isoformat(), anchor=str(anchor), asof=str(t_last.date()), exec_date=str(e.date()), status='pending')
    if rows:
        DEC = pd.concat([DEC, pd.DataFrame(rows).reindex(columns=DEC_COLS)], ignore_index=True)
    if not stale and not already:
        S['last_exec'] = str(e.date())
    S['last_clock'] = clock.isoformat()
    DEC = fill_exec(DEC, data, dsp, cal, D, S)
    DEC.to_csv(dec_f, index=False)
    st_f.write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')
    nav_monthly(DEC, data, cal, S, OUT, Path(a.e50))
    run['n_rows'] = len(rows)
    with open(OUT / 'ec_runs.jsonl', 'a', encoding='utf-8') as f: f.write(json.dumps(run, ensure_ascii=False) + '\n')
    print(json.dumps(run, ensure_ascii=False))


def fill_exec(DEC, data, dsp, cal, D, S):
    """補成交日的開盤、漲跌停、處置、量能、K5T 估計；遇漲停買進／跌停或停牌賣出 → blocked，並修正持股狀態。"""
    if not len(DEC): return DEC
    C, O, V, H = data['C'], data['O'], data['V'], data['H']
    hcal = data['cal']
    odd = None
    op = D / 'extras/oddlot_hist.csv.gz'
    if op.exists():
        try:
            odd = pd.read_csv(op, dtype={'code': str}, parse_dates=['date']).set_index(['date', 'code'])['shares']
        except Exception:
            odd = None
    gcal = hcal[(C.notna().sum(axis=1) >= 300).to_numpy()]
    for i, r in DEC[DEC.status == 'pending'].iterrows():
        e = pd.Timestamp(r.exec_date); c = r.code
        if e not in hcal: continue
        k = hcal.get_loc(e)
        pc = C[c].iloc[:k].dropna().iloc[-1] if c in C and C[c].iloc[:k].notna().any() else np.nan
        o = O[c].get(e, np.nan) if c in O else np.nan; cl = C[c].get(e, np.nan) if c in C else np.nan
        lu = MP.limit_up_price(np.array([pc]))[0] if np.isfinite(pc) else np.nan
        ld = math.ceil(pc * 0.9 / tick(pc * 0.9) - 1e-9) * tick(pc * 0.9) if np.isfinite(pc) else np.nan
        gk = gcal[gcal < e]
        v = V[c].reindex(gk[-60:]) if c in V else pd.Series(dtype=float)
        v60 = float(v.mean()) if len(v) and v.notna().sum() >= 40 else np.nan
        rr = (C[c].reindex(gk[-61:]).pct_change()) if c in C else pd.Series(dtype=float)
        sg = float(rr.std()) if rr.notna().sum() >= 40 else np.nan
        allv = V.reindex(gk[-60:]).mean()
        px_last = C.reindex(gk[-1:]).iloc[0]
        base = (allv >= 1e7) & (px_last >= 10)
        vr = allv.where(base).rank(ascending=False).get(c, np.nan)
        dec = int(min(10, max(1, math.ceil(vr / 30)))) if np.isfinite(vr) and vr <= 300 else 10
        oa = OASH[dec - 1] * v60 if np.isfinite(v60) else np.nan
        side = 1 if r.side == 'buy' else -1
        indisp = bool(((dsp['code'] == c) & (dsp['start'] <= e) & (dsp['end'] >= e)).any())
        blocked = None
        if not np.isfinite(o): blocked = 'no_open'
        elif side > 0 and np.isfinite(lu) and o >= lu - 1e-6: blocked = 'open_limit_up'
        elif side < 0 and np.isfinite(ld) and o <= ld + 1e-6: blocked = 'open_limit_down'
        upd = dict(open=o, prev_close=pc, limit_up=lu, limit_dn=ld, open_limit=int(blocked in ('open_limit_up', 'open_limit_down')), close=cl,
                   in_disposal=int(indisp), sigma60=sg, v60=v60, dec=dec, oa_pred=oa,
                   lots_traded=float(V[c].get(e, np.nan) / cl / 1000) if c in V and np.isfinite(cl) and cl > 0 else np.nan,
                   odd_shares_traded=float(odd.get((e, c), np.nan)) if odd is not None else np.nan)
        for tier, tag in zip(TIERS, ('100w', '300w', '1000w')):
            Q = tier / N_HOLD
            if np.isfinite(o) and o > 0:
                L = int(Q // (1000 * o)); q = int((Q - L * 1000 * o) // o)
                imp = max(0.001 + 0.5 * (sg if np.isfinite(sg) else 0.03) * math.sqrt(L * 1000 * o / oa), tick(o) / o) if (L > 0 and np.isfinite(oa) and oa > 0) else np.nan
                upd.update({f'lots_{tag}': L, f'odd_{tag}': q, f'k5t_{tag}': imp})
        upd['status'] = f'blocked_{blocked}' if blocked else 'filled'
        for kk, vv in upd.items(): DEC.at[i, kk] = vv
        if blocked:
            port = 'ec' if r.portfolio == 'EC' else 'off'
            if side > 0 and c in S[port]: S[port].remove(c)
            if side < 0 and c not in S[port]: S[port].append(c)
    return DEC


def nav_monthly(DEC, data, cal, S, OUT, e50p):
    """由成交紀錄重算 EC／OFF 紙上淨值（100 萬級距、一般與壓力成本）與官方訊號日到訊號日的月配對差。"""
    C, O = data['C'], data['O']; ADJ = data['ADJ']; hcal = data['cal']
    t0 = pd.Timestamp(S['init_exec'])
    if t0 not in hcal: return
    days = hcal[hcal >= t0]
    AO = (ADJ * O / C)
    fills = DEC[DEC.status == 'filled'].copy()
    out = {}
    for cost in ('normal', 'stress'):
        for port in ('EC', 'OFF'):
            units = {}; cash = 0.0; nav = []
            for c in S['init_list']:
                px = AO[c].get(t0, np.nan) if c in AO else np.nan
                if np.isfinite(px): units[c] = 0.1 / px
                else: cash += 0.1
            for d in days:
                if d > t0:
                    f = fills[(fills.portfolio == port) & (pd.to_datetime(fills.exec_date) == d)]
                    for _, r in f[f.side == 'sell'].iterrows():
                        if r.code not in units: continue
                        px = AO[r.code].get(d, np.nan)
                        if not np.isfinite(px): continue
                        imp = r.k5t_100w if np.isfinite(r.k5t_100w) else 0.003
                        cst = (FEE_S if cost == 'stress' else FEE_N) + imp + TAX + (EXTRA_S if cost == 'stress' else 0)
                        cash += units.pop(r.code) * px * (1 - cst)
                    b = f[f.side == 'buy']
                    if len(b) and cash > 0:
                        each = cash / len(b)
                        for _, r in b.iterrows():
                            px = AO[r.code].get(d, np.nan)
                            if not np.isfinite(px): continue
                            imp = r.k5t_100w if np.isfinite(r.k5t_100w) else 0.003
                            cst = (FEE_S if cost == 'stress' else FEE_N) + imp + (EXTRA_S if cost == 'stress' else 0)
                            units[r.code] = units.get(r.code, 0) + each * (1 - cst) / px; cash -= each
                val = cash + sum(u * (ADJ[c].get(d, np.nan) if np.isfinite(ADJ[c].get(d, np.nan)) else ADJ[c].loc[:d].dropna().iloc[-1]) for c, u in units.items())
                nav.append(val)
            out[f'{port}_{cost}'] = pd.Series(nav, index=days)
    N = pd.DataFrame(out)
    try:
        e5 = pd.read_csv(e50p, parse_dates=['date']).set_index('date')['adj'].reindex(days).ffill()
        N['E0050'] = e5 / e5.iloc[0]
    except Exception:
        N['E0050'] = np.nan
    N.index.name = 'date'; N.to_csv(OUT / 'ec_nav.csv')
    # 月配對：官方訊號日收盤到下一個官方訊號日收盤
    sigs = []
    m = pd.Period(t0, 'M')
    while True:
        s = cal.signal_day(m)
        if s > days[-1]: break
        if s in N.index: sigs.append(s)
        m += 1
    rows = []
    for a, b in zip(sigs[:-1], sigs[1:]):
        r = {k: N.loc[b, k] / N.loc[a, k] - 1 for k in N.columns}
        rows.append(dict(anchor=str(pd.Period(b, 'M')), start=str(a.date()), end=str(b.date()),
                         ec_normal=r['EC_normal'], off_normal=r['OFF_normal'], diff_normal=r['EC_normal'] - r['OFF_normal'],
                         ec_stress=r['EC_stress'], off_stress=r['OFF_stress'], diff_stress=r['EC_stress'] - r['OFF_stress'], r0050=r['E0050']))
    pd.DataFrame(rows).to_csv(OUT / 'ec_monthly.csv', index=False)


if __name__ == '__main__':
    main()
