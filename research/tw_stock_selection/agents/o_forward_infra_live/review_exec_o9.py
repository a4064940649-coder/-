#!/usr/bin/env python3
"""反方審查（成本與可成交、資料實際取得時點）— o_forward_infra_live。只讀研究員檔案；暫存寫到 scratchpad。
用法：python3 -I -B review_exec_o9.py <scratch_dir>
  <scratch_dir>/v1012/ 需先放 2026-10-12 晚（commit 50bda154，台北 10-13 03:48）上游 extras 的 rev_hist、warnings_hist、disposal_5y。
"""
import sys, json, math, subprocess, re
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
TW = HERE.parent.parent
sys.path.insert(0, str(TW)); sys.path.insert(0, str(HERE))
import fwd_data as FD          # noqa
import disp_sat as DS          # noqa
import main_picks_today as MP  # noqa

SCR = Path(sys.argv[1]); OUT = {}
SRC = HERE / 'raw/screener/data'
git = lambda *a: subprocess.run(['git', '-C', str(HERE / 'raw/screener'), *a], capture_output=True, text=True).stdout.strip()

# ---------------- R1 上游資料在重播期是否存在（實際取得時點） ----------------
first = {}
for f in ['data/history.csv.gz', 'data/extras/rev_hist.csv.gz', 'data/extras/warnings_hist.csv.gz', 'data/extras/rev_5y.csv.gz',
          'data/extras/disposal_5y.csv.gz']:
    lg = git('log', '--reverse', '--format=%cI', '--', f).splitlines()
    first[f] = dict(first_commit=lg[0] if lg else None, last_commit=lg[-1] if lg else None, n=len(lg))
first['repo_first_commit'] = git('log', '--reverse', '--format=%cI').splitlines()[0]
OUT['R1_upstream_first_commit'] = first

# ---------------- R2 2026-10-12 官方日：重播（今日版本截斷）vs 當晚真實版本 ----------------
v = SCR / 'v1012'
src = FD.Source()
asof = pd.Timestamp('2026-10-12')
cur_rh = src._rev_trunc(src.rh, asof); cur_rh = cur_rh[cur_rh['ym'] == '2026-09']
vin_rh = pd.read_csv(v / 'rev_hist.csv.gz', dtype={'code': str}); vin_rh = vin_rh[vin_rh['ym'] == '2026-09']
cs, vs = set(cur_rh['code']), set(vin_rh['code'])
cur_d = src.disposal(); cur_d = cur_d[pd.to_datetime(cur_d['date']) <= asof]
vw = pd.read_csv(v / 'warnings_hist.csv.gz', dtype={'code': str}); vd5 = pd.read_csv(v / 'disposal_5y.csv.gz', dtype={'code': str})
vdisp = pd.concat([vd5[['date', 'code', 'start', 'end']], vw[vw['flag'] == '處置'][['date', 'code', 'start', 'end']]])
key = lambda d: set(zip(d['code'], d['start'].astype(str), d['end'].astype(str)))
only_cur = key(cur_d) - key(vdisp)
recent_only = sorted(x for x in only_cur if x[1] >= '2026-09-01')
OUT['R2_vintage_20261012'] = dict(
    rev_2026_09_rows_in_replay_trunc=len(cs), rev_2026_09_rows_in_vintage=len(vs), in_replay_not_in_vintage=len(cs - vs),
    in_vintage_not_in_replay=len(vs - cs),
    warnings_hist_rows_vintage=len(vw), warnings_hist_rows_now=len(src.wh), warnings_hist_min_date_vintage=str(vw['date'].min()),
    disposal_events_le_1012_replay=len(key(cur_d)), disposal_events_le_1012_vintage=len(key(vdisp)),
    events_since_0901_in_replay_not_vintage=len(recent_only), sample=recent_only[:15],
    has_2466_6213_vintage=sorted(x for x in key(vdisp) if x[0] in ('2466', '6213') and x[1] >= '2026-10-01'))
# 受影響的處置事件是否落在 A5 的 101 筆
led = pd.read_csv(HERE / 'replay/seq/disposal_ledger.csv', dtype=str)
led_r = led[(led['announce'] >= '2026-09-01') & (led['announce'] <= '2026-11-02')]
OUT['R2b_A5_events'] = dict(n_replay_events_0901_1102=len(led_r), ts_ok=int(led_r['ts_ok'].astype(int).sum()),
                            announced_before_warnings_hist_existed=int((led_r['announce'] < '2026-10-06').sum()),
                            first_seen_minus_announce_hours=sorted(set(((pd.to_datetime(led_r['first_seen_ts']) - pd.to_datetime(led_r['announce_ts'])).dt.total_seconds() / 3600).round(1)))[:10])

# ---------------- R3 排程路徑的資料過期偵測 ----------------
code = (HERE / 'make_forward.py').read_text()
m1 = re.search(r"want = pd.Timestamp\(a.asof\) if a.asof else src.hcal\[-1\]", code) is not None
sh = (HERE / 'daily_step0.sh').read_text()
OUT['R3_stale_in_scheduled_path'] = dict(want_defaults_to_last_data_day=m1, daily_step0_passes_asof=('--asof' in sh),
                                          conclusion='排程路徑不帶 --asof → want＝資料最後交易日 → want>hcal[-1] 永遠不成立 → 上游延遲時回傳 0，不是 2')

# ---------------- R4 處置衛星：資金級距與事前不可知資訊 ----------------
fr_dir = SCR / 'dspdir'; (fr_dir / 'extras').mkdir(parents=True, exist_ok=True)
for a, b in [('history.csv.gz', SRC / 'history.csv.gz'), ('stock_list.csv', SRC / 'stock_list.csv'),
             ('extras/exdiv_5y.csv.gz', SRC / 'extras/exdiv_5y.csv.gz'), ('taiex_twii.csv', FD.TAIEX_SRC), ('etf_0050.csv', FD.E50_SRC)]:
    p = fr_dir / a
    if not p.exists(): p.symlink_to(b)
FD.write_csv(src.disposal().drop(columns='_src'), fr_dir / 'extras/disposal_5y.csv.gz')
fr = DS.load_frames(fr_dir)
ev = DS.events(fr)
ledger = led  # 重播帳本的 ts_ok


def sim(equity, no_close_filter=False, ex_ante_only=False, skip_second=False, line='DISP_TWSE'):
    """disp_sat.simulate 的複本（DISP_TWSE），只加開關：no_close_filter＝不用 s 收盤漲幅（事前不可知）；
    ex_ante_only＝不用 s 當日成交值 V_s 縮單（事前不可知，只用 V60(t−1)）。"""
    cal = fr['cal']; days = cal[cal >= pd.Timestamp('2026-09-01')]
    C, R, V, V60p, S60p, PC = fr['C'], fr['R'], fr['V'], fr['V60p'], fr['S60p'], fr['PC']
    e50 = fr['e50'].reindex(cal.union(fr['e50'].index)).ffill().reindex(cal); tx = fr['tx']
    okts = dict(zip(ledger['event_id'], ledger['ts_ok'].astype(int))); E = ev.set_index('event_id')
    fee = DS.FEE_N
    cand = ev[(ev['market'] == 'TWSE') & ev['chain_take'].astype(bool)]
    if skip_second: cand = cand[~cand['second'].astype(bool)]
    cand = cand[cand['s'].notna() & (cand['s'] >= days[0])]
    by_s = {s: g for s, g in cand.groupby('s')}
    idle = float(equity); pos = {}; nav = {days[0]: idle}; n_buy = 0; capped = 0; lim_fill = 0; amts = []
    for k, t in enumerate(days[1:], start=1):
        tp = days[k - 1]
        idle *= e50[t] / e50[tp] if np.isfinite(e50[t]) and np.isfinite(e50[tp]) else 1.0
        for p in pos.values():
            r = R.at[t, p['code']]
            if np.isfinite(C.at[t, p['code']]): p['val'] *= (1 + (r if np.isfinite(r) else 0.0))
        for eid in list(pos):
            p = pos[eid]; c = p['code']
            if not (t >= p['e']): continue
            px, r = C.at[t, c], R.at[t, c]
            if not np.isfinite(px) or (np.isfinite(r) and r <= -0.09): continue
            inD = t <= p['e']
            qref = (V60p.at[t, c] if ex_ante_only else V.at[t, c]) * (p['mm'] / 270.0 if inD else 0.25)
            sh_cap = math.floor(DS.CAP * qref / px) if np.isfinite(qref) else 0
            n = min(p['sh'], max(sh_cap, 0))
            if n <= 0: continue
            sg = S60p.at[t, c]; q = n * px
            imp = max(float(DS.tick(px)) / px, DS.BASE) + (0.5 * sg * math.sqrt(q / qref) if np.isfinite(sg) and qref > 0 else 0.0)
            fpx = px * (1 - imp); vv = n * fpx; fe = max(20.0, fee * vv)
            idle += (vv - fe - DS.TAX * vv) * (1 - fee)
            frac = n / p['sh']; p['val'] *= (1 - frac); p['sh'] -= n
            if p['sh'] <= 0: del pos[eid]
        if t in by_s:
            txp = tx[tx.index < t]
            bear = bool(len(txp) >= 200 and txp.iloc[-1] < txp.iloc[-200:].mean())
            for eid, r_ in by_s[t].sort_values(['announce', 'code']).set_index('event_id').iterrows():
                c = r_['code']
                if okts.get(eid, 0) != 1 or bear or any(p['code'] == c for p in pos.values()) or len(pos) >= DS.SLOTS: continue
                px = C.at[t, c]; r = R.at[t, c]
                if not np.isfinite(px): continue
                if np.isfinite(r) and r >= 0.09:
                    if not no_close_filter: continue
                    lim_fill += 1
                v60, pc = V60p.at[t, c], PC.at[t, c]
                if not (np.isfinite(v60) and v60 >= 1e7 and np.isfinite(pc) and pc >= 10): continue
                navp = idle + sum(p['val'] for p in pos.values()); mm = int(r_['match_min'])
                slot = navp / DS.SLOTS
                amt = min(slot, DS.CAP * v60 * mm / 270.0)
                qref = (v60 if ex_ante_only else V.at[t, c]) * mm / 270.0
                if not ex_ante_only: amt = min(amt, DS.CAP * qref) if np.isfinite(qref) else 0.0
                if amt < 0.95 * slot: capped += 1
                sg = S60p.at[t, c]
                imp = max(float(DS.tick(px)) / px, DS.BASE) + (0.5 * sg * math.sqrt(amt / qref) if np.isfinite(sg) and qref > 0 else 0.0)
                fpx = px * (1 + imp); n = math.floor(amt / fpx)
                if n <= 0: continue
                vv = n * fpx; fe = max(20.0, fee * vv); need = (vv + fe) / (1 - fee - DS.ETF_TAX)
                if need > idle: continue
                idle -= need; n_buy += 1; amts.append(vv / slot)
                pos[eid] = dict(code=c, sh=n, val=n * px, e=E.at[eid, 'e'] if pd.notna(E.at[eid, 'e']) else pd.Timestamp('2100-01-01'), mm=mm)
        for eid, p in pos.items():
            if p['e'].year == 2100 and pd.notna(E.at[eid, 'e']): p['e'] = E.at[eid, 'e']
        nav[t] = idle + sum(p['val'] for p in pos.values())
    nv = pd.Series(nav)
    def win(a, b):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        r = nv[nv.index <= b].iloc[-1] / nv[nv.index <= a].iloc[-1] - 1
        r5 = e50[e50.index <= b].iloc[-1] / e50[e50.index <= a].iloc[-1] - 1
        return round((r - r5) * 100, 3)
    return dict(equity=equity, n_buy=n_buy, n_capped=capped, median_fill_over_slot=round(float(np.median(amts)), 3) if amts else None,
                limitup_close_bought=lim_fill, ex_0911_1012=win('2026-09-11', '2026-10-12'), ex_1012_1102=win('2026-10-12', '2026-11-02'),
                ex_0901_1102=win('2026-09-01', '2026-11-02'))

rep = pd.read_csv(HERE / 'replay/seq/forward_returns.csv')
OUT['R4_replay_DISP_TWSE'] = rep[rep.line.isin(['DISP_TWSE', 'DISP_TWSE_FIRST'])][['line', 'start', 'end', 'excess_pp']].to_dict('records')
R4 = {}
for eq in (300_000, 1_000_000, 3_000_000, 10_000_000):
    R4[f'base_{eq}'] = sim(eq)
R4['noclosefilter_1e6'] = sim(1_000_000, no_close_filter=True)
R4['exante_1e6'] = sim(1_000_000, ex_ante_only=True)
R4['exante_noclose_1e6'] = sim(1_000_000, no_close_filter=True, ex_ante_only=True)
R4['exante_noclose_1e7'] = sim(10_000_000, no_close_filter=True, ex_ante_only=True)
R4['skip_second_1e6'] = sim(1_000_000, skip_second=True)
OUT['R4_disp_sensitivity'] = R4
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))
