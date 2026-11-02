#!/usr/bin/env python3
"""2026-11 起前瞻追蹤產生器 v3（o_forward_infra_live；修正 o_forward_sequential_v2/make_forward.py 的 o8 瑕疵，原檔不動）。

修正（對 o8 審查）
- F-A 訊號日當晚凍結：名單快照只需要訊號日（含）以前的資料；隔日開盤資料不存在時照樣寫入完整排序（全部合格股），
  成交日再依凍結排序套用 M2（停牌不買、開盤漲停且未持有不買、由下一名遞補）。快照第一次寫入後不覆寫；
  之後重算（最近 RECHECK 個交易日內的訊號日）若前 20 名不同 → list_diffs.csv。
- F-B --asof 截斷全部輸入：fwd_data.Source.build_asof 寫出截斷資料夾（價格、營收 ann_eff ≤ asof、處置公告日 ≤ asof、0050、加權指數、面板），
  載入後以 fwd_data.assert_asof／assert_dir_asof 斷言每個資料框最大日期 ≤ asof；每個訊號日的名單用「截斷到該訊號日」的資料算。
- F-C J0 的 ann_date 缺值一律視為未公告（沿用 m−2 的 3 月年增率）；J≥1 缺值視為已公告（法定期限已過）。
  任何 J 線若 m−1 月營收在截斷資料中不存在，一律沿用 m−2（舊版會變成缺值而剔除）。
- F-D 成本＝V7X K5T：xlive.py 逐字載入 o_exec_engine_rewrite_v7x/x_exec.py（零股、2% 現金、參與上限 20%、處置期收盤減一跳賣出、
  開盤漲跌停鎖死整股零股都不成交順延、F1～F4），GROW 100 萬（2026 幣值）；另報壓力成本與參與上限 10%（SOP）。
名單規格：與 main_picks_today.py（sha256 0897787f…，2026-11-02 版）--ex-fin 相同；官方日線直接呼叫 MP.score_on／MP.execute。
時程：錨定月 m 的 D_m＝m 月 10 日起第一個交易日＝J0；J{j}＝D_m 之後第 j 個交易日；官方日＝第一個 ≥ 11 日的交易日。
用法（每日第 0 步，資料 git pull 之後）：
  python3 -I -B make_forward.py [--asof YYYY-MM-DD] [--out <live_tracking/main_forward>]
重播（驗收）：--replay --sim-clock 2026-09-11T06:49+08:00 --out <replay 目錄> --start 2026-09
"""
import argparse, json, hashlib, sys, datetime, subprocess, math, shutil
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TW = AG.parent
sys.path.insert(0, str(TW))
import main_picks_today as MP   # noqa: E402（只讀 import）
sys.path.pop(0)
sys.path.insert(0, str(HERE))
import fwd_data as FD           # noqa: E402
import xlive as XL              # noqa: E402
import disp_sat as DS           # noqa: E402

NJ, NPICK, TOPCHK = 13, 10, 20
TZ = 'Asia/Taipei'
LIVE_OUT = AG / 'live_tracking/main_forward'
MIX_REB_COST = 0.005
G8_LINES = ['JMED', 'DISP_TWSE', 'MIX_C70', 'MIX_D80']
FR_COLS = ['line', 'anchor', 'start', 'end', 'status', 'n_j', 'ret', 'ret_0050', 'excess_pp', 'ret_stress', 'excess_stress_pp',
           'ret_cap10', 'excess_cap10_pp', 'cost', 'n_hold', 'source', 'note']
TR_COLS = ['kind', 'line', 'date', 'code', 'side', 'is_odd', 'shares', 'fill_px', 'open_px', 'prev_close', 'close_px', 'ticks_from_open',
           'sigma60_prev', 'V60_prev', 'dec', 'oa_share_dec', 'OA_pred', 'share_of_open_vol', 'share_of_open_vol_actual', 'in_disposal',
           'event_id', 'cost_rate', 'share_of_qref', 'order_time', 'limit_px']
FILL_FIELDS = ['date', 'code', 'side', 'order_time', 'limit_px', 'fill_px', 'shares', 'is_odd', 'list']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ts_str(t):
    return pd.Timestamp(t).tz_convert(TZ).isoformat()


# ---------------------------------------------------------------- 資料視圖
class View:
    """截斷到 d 的資料（MP.load_prices 的回傳＋營收＋處置＋公告日表）。"""
    def __init__(self, src, d, work, keep=False):
        self.d = pd.Timestamp(d)
        self.dir = Path(work) / f'asof_{self.d:%Y%m%d}'
        self.man = src.build_asof(self.d, self.dir)
        self.man_sha = sha(self.dir / 'MANIFEST.json')
        pnl = src.panel_pre(work)
        self.dir_check = {**FD.assert_dir_asof(self.dir, self.d), **FD.assert_panel_asof(pnl, self.d)}
        self.data = MP.load_prices(self.dir, self.d, pnl)
        self.YOY, self.YOY3, _ = MP.load_revenue(self.dir)
        self.dsp = MP.load_disposal(self.dir)
        r = pd.read_csv(self.dir / 'extras/rev_hist.csv.gz', dtype={'code': str})
        r = r[r['ann_date'].notna()]
        self.ann = {(c, y): pd.Timestamp(a) for c, y, a in zip(r.code, r.ym, r.ann_date)}
        self.e50 = pd.read_csv(self.dir / 'etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
        dd = self.data
        self.loaded = FD.assert_asof(self.d, C=dd['C'], O=dd['O'], H=dd['H'], V=dd['V'], R=dd['R'], ADJ=dd['ADJ'],
                                     disposal=self.dsp, e50=self.e50, ann=pd.DataFrame({'date': list(self.ann.values())}))
        assert self.YOY.index.max() <= pd.Period(self.d, 'M') - 1, 'A1 失敗：營收月份超過 asof 前一月'
        self.loaded['revenue_max_ym'] = str(self.YOY.index.max())
        if not keep: self.drop()

    def drop(self):
        shutil.rmtree(self.dir, ignore_errors=True)       # 資料已在記憶體；磁碟只留 MANIFEST 的 sha（省空間）


def month_days(cal, m):
    c = cal[(cal.year == m.year) & (cal.month == m.month) & (cal.day >= 10)]
    if not len(c): return None
    k = cal.get_loc(c[0])
    return dict(D=c[0], J=[cal[k + j] if k + j < len(cal) else None for j in range(NJ)], official=MP.rebalance_day(cal, m.year, m.month))


def is_fin(data, cols):
    return pd.Series(cols.map(lambda c: data['ind'].get(c, '') == MP.FIN_IND or c[:2] == '28' or c in ('5876', '5880')), index=cols)


def rank_official(v, sd):
    order, full, m, n_el = MP.score_on(v.data, v.YOY3, v.dsp, sd, ex_fin=True)
    full = full.assign(rev_month=str(m))
    return order, full, {}


def rank_j(v, sd, anchor, j):
    """J 線：同 MP.score_on，但 m−1 月營收只在已公告時使用（J0 缺 ann_date＝未公告；J≥1 缺＝已公告）。"""
    _, df, _, _ = MP.score_on(v.data, v.YOY3, v.dsp, sd, ex_fin=True)
    cols = df.index
    pm1, pm2 = anchor - 1, anchor - 2
    new = v.YOY3.loc[pm1].reindex(cols) if pm1 in v.YOY3.index else pd.Series(np.nan, index=cols)
    old = v.YOY3.loc[pm2].reindex(cols) if pm2 in v.YOY3.index else pd.Series(np.nan, index=cols)
    a = pd.Series([v.ann.get((c, str(pm1)), pd.NaT) for c in cols], index=cols)
    miss = a.isna()
    ok_new = (a <= sd).fillna(False).astype(bool) | (miss & (j > 0))
    ok = ok_new & new.notna()
    ok_oldrule = ((a <= sd).fillna(False).astype(bool) | miss) & new.notna()        # o8 版：缺值視為已公告
    finm = is_fin(v.data, cols)

    def build(okm):
        y3 = new.where(okm, old)
        el = df['t300'] & y3.notna() & df['s52'].notna() & ~df['inD'] & ~finm
        sc = 0.5 * df['s52'].where(el).rank(pct=True) + 0.5 * y3.where(el).rank(pct=True)
        full = df.assign(yoy3=y3, el=el, score=sc, rev_month=np.where(okm, str(pm1), str(pm2)))
        full['score_r'] = full['score'].round(10)
        order = full[full['el']].reset_index().sort_values(['score_r', 'yoy3', 'code'], ascending=[False, False, True]).drop(columns='score_r').set_index('code')
        return order, full
    order, full = build(ok)
    info = {}
    if j == 0:
        o_old, _ = build(ok_oldrule)
        aff = sorted(c for c in cols[(miss & new.notna() & df['t300'] & ~finm).to_numpy()])
        info = dict(n_ann_missing_pool=len(aff), affected_sample=aff[:30], top10_new_rule=list(order.index[:NPICK]),
                    top10_old_rule=list(o_old.index[:NPICK]), n_top10_changed=len(set(o_old.index[:NPICK]) ^ set(order.index[:NPICK])) // 2)
    return order, full, info


def snapshot_frame(v, order, full, line, m, sd):
    names = v.data['names']
    rows = []
    for k, code in enumerate(order.index):
        r = full.loc[code]
        rows.append(dict(line=line, anchor=str(m), signal=str(sd.date()), rank=k + 1, code=code, name=names.get(code, ''),
                         score=float(r['score']), s52=float(r['s52']), yoy3=float(r['yoy3']), rev_month=str(r.get('rev_month', '')),
                         close=float(r['close']), v60=float(r['v60'])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(FD.SRC))
    ap.add_argument('--e50', default=str(FD.E50_SRC))
    ap.add_argument('--taiex', default=str(FD.TAIEX_SRC))
    ap.add_argument('--panel', default=str(FD.PANEL_SRC))
    ap.add_argument('--out', default=str(LIVE_OUT))
    ap.add_argument('--work', default=str(HERE / 'work'))
    ap.add_argument('--start', default='2026-11')
    ap.add_argument('--disp-start', default=None, help='處置衛星起算日（預設＝start 月第一個交易日）')
    ap.add_argument('--asof')
    ap.add_argument('--replay', action='store_true', help='重播：時戳用 --sim-clock，不要求 PREREG')
    ap.add_argument('--sim-clock')
    ap.add_argument('--recheck', type=int, default=20, help='重算比對最近幾個交易日的訊號日名單')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pl_dir = out / 'paper_lists'; pl_dir.mkdir(exist_ok=True)
    if a.replay:
        assert a.sim_clock, '--replay 需要 --sim-clock'
        clock, ckind = pd.Timestamp(a.sim_clock).tz_convert(TZ), 'sim'
    else:
        clock, ckind = pd.Timestamp.now(tz=TZ), 'wall'
        assert out.resolve() != (HERE / 'replay').resolve()
    prereg = None
    rj = HERE / 'results.json'
    if rj.exists(): prereg = json.loads(rj.read_text(encoding='utf-8')).get('PREREG_FORWARD')
    if not a.replay:
        assert prereg is not None, 'PREREG_FORWARD 不存在：凍結前不得寫入正式前瞻資料夾（只能 --replay）'
    src = FD.Source(a.src, a.e50, a.taiex, a.panel)
    head = ''
    try:
        head = subprocess.run(['git', '-C', str(Path(a.src).parent), 'log', '-1', '--format=%H %cI'], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    rec = dict(clock=clock.isoformat(), clock_kind=ckind, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), data_head=head,
               script_sha256=sha(__file__), lib_sha256={p: sha(HERE / p) for p in ('fwd_data.py', 'xlive.py', 'disp_sat.py')},
               mp_sha256=sha(TW / 'main_picks_today.py'), engine_sha256=sha(XL.XEXEC), src_last=str(src.hcal[-1].date()))
    if prereg:
        reg = prereg.get('file_sha256', {})
        cur = {'make_forward.py': rec['script_sha256'], **rec['lib_sha256'], 'main_picks_today.py': rec['mp_sha256'], 'x_exec.py': rec['engine_sha256']}
        rec['sha_matches_prereg'] = {k: (reg.get(k) == v) for k, v in cur.items()}
    want = pd.Timestamp(a.asof) if a.asof else src.hcal[-1]
    if want > src.hcal[-1]:
        if src.is_trading(want) or want - src.hcal[-1] > pd.Timedelta(days=4):
            rec.update(status='DATA_STALE', asof=str(want.date()),
                       note=f'來源資料只到 {src.hcal[-1].date()}，asof {want.date()} 的資料不存在；不寫任何名單（請先 git pull）')
            with open(out / 'runs.jsonl', 'a', encoding='utf-8') as fh: fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(json.dumps(rec, ensure_ascii=False)); sys.exit(2)
    asof = src.hcal[src.hcal <= want][-1]
    rec['asof'] = str(asof.date())
    work = Path(a.work)
    V = View(src, asof, work, keep=True)
    rec['loaded_max_dates'] = V.loaded; rec['dir_check'] = V.dir_check; rec['asof_manifest_sha'] = V.man_sha
    cal = V.data['cal']
    start = pd.Period(a.start, 'M')
    months = [m for m in pd.period_range(start, pd.Period(asof, 'M'), freq='M')]
    MD = {m: month_days(cal, m) for m in months}
    diffs, frozen_now, views = [], [], {asof: V}

    def view(d):
        if d not in views:
            for k in [k for k in views if k != asof]: del views[k]     # 只留一個舊視圖，控制記憶體
            views[d] = View(src, d, work)
        return views[d]
    # ---- 1. 名單快照（訊號日當晚凍結） ----
    recent = set(cal[-a.recheck:])
    for m in months:
        md = MD[m]
        if md is None: continue
        days = [('official', md['official'])] + [(f'j{j}', md['J'][j]) for j in range(NJ)]
        for line, sd in days:
            if sd is None or sd > asof: continue
            snap = pl_dir / f'{line}_{m.strftime("%Y%m")}.csv'; meta = snap.with_suffix('.meta.json')
            if snap.exists() and sd not in recent: continue
            v = view(sd)
            if line == 'official': order, full, info = rank_official(v, sd)
            else: order, full, info = rank_j(v, sd, m, int(line[1:]))
            tab = snapshot_frame(v, order, full, line, m, sd)
            if snap.exists():
                old = pd.read_csv(snap, dtype={'code': str})
                o1, o2 = list(old.sort_values('rank').code[:TOPCHK]), list(tab.code[:TOPCHK])
                if o1 != o2:
                    diffs.append(dict(kind='list', line=line, anchor=str(m), signal=str(sd.date()), asof=str(asof.date()),
                                      snapshot=','.join(o1), recomputed=','.join(o2)))
                continue
            exec_est = src.next_trading(sd)
            open_ts = pd.Timestamp(exec_est.date()).tz_localize(TZ) + pd.Timedelta(hours=9)
            sha_tab = FD.write_csv(tab, snap, gz=False)
            mt = dict(line=line, anchor=str(m), signal=str(sd.date()), freeze_asof=str(asof.date()), freeze_on_signal_day=bool(sd == asof),
                      frozen_ts=clock.isoformat(), clock_kind=ckind, wall_utc=rec['utc'], exec_day_est=str(exec_est.date()),
                      exec_open_ts=open_ts.isoformat(), ts_ok=bool(clock < open_ts), data_head=head, view_manifest_sha=v.man_sha,
                      view_max_dates=v.loaded, snapshot_sha256=sha_tab, script_sha256=rec['script_sha256'], mp_sha256=rec['mp_sha256'],
                      n_eligible=int(len(tab)), top10=list(tab.code[:NPICK]), j0_ann_info=info)
            meta.write_text(json.dumps(mt, ensure_ascii=False, indent=1, sort_keys=True))
            frozen_now.append(f'{line}_{m.strftime("%Y%m")}')
    views = {asof: V}
    # ---- 2. 成交（隔日開盤，M2 遞補，依凍結排序） ----
    lines = ['official'] + [f'j{j}' for j in range(NJ)]
    ent = {ln: [] for ln in lines}
    for ln in lines:
        held = set()
        for m in months:
            snap = pl_dir / f'{ln}_{m.strftime("%Y%m")}.csv'
            if not snap.exists(): continue
            tab = pd.read_csv(snap, dtype={'code': str}).sort_values('rank')
            sd = pd.Timestamp(tab.signal.iloc[0])
            exf = snap.with_suffix('.exec.json')
            if exf.exists():
                ex = json.loads(exf.read_text())
            else:
                if sd not in cal or cal.get_loc(sd) + 1 >= len(cal): ent[ln].append((m, sd, None, None)); continue
                r = MP.execute(V.data, pd.DataFrame(index=pd.Index(list(tab.code), name='code')), sd, NPICK, held)
                ex = dict(signal=str(sd.date()), exec=str(r['next_day'].date()), picks=r['picks'], skipped=[list(x) for x in r['skipped']],
                          held_before=sorted(held), computed_asof=str(asof.date()), computed_ts=clock.isoformat())
                exf.write_text(json.dumps(ex, ensure_ascii=False, indent=1, sort_keys=True))
            ent[ln].append((m, sd, pd.Timestamp(ex['exec']), ex['picks']))
            held = set(ex['picks'])
    # ---- 3. V7X K5T 紙上淨值 ----
    e50 = V.e50
    first_sd = min([e[1] for v_ in ent.values() for e in v_] or [asof])
    kstart = cal[max(cal.searchsorted(first_sd) - 130, 0)]
    dmod = XL.build_module(V.data, V.dsp, e50, start=kstart)
    XE = XL.load_engine(dmod); LSim = XL.logging_sim(XE)
    navs, fills = {}, []
    for ln in lines:
        pl = [(dmod.TPOS[nd], np.array([dmod.CI[c] for c in pk], int), dmod.TPOS[sd]) for (m, sd, nd, pk) in ent[ln] if nd is not None]
        if not pl: continue
        for var in ('normal', 'stress', 'cap10'):
            cfg = dict(XL.CFG_V7X, cost='stress' if var == 'stress' else 'normal')
            cap0 = XE.CAP
            if var == 'cap10': XE.CAP = 0.10
            try:
                s = LSim(pl, cfg); nav = s.run()
            finally:
                XE.CAP = cap0
            navs[(ln, var)] = nav
            if var == 'normal':
                for f in s.fills: fills.append(dict(f, line=ln.upper(), kind='paper_' + f['kind']))
    E = e50

    def at(s, d):
        s = s.dropna(); k = s.index.searchsorted(pd.Timestamp(d), side='right') - 1
        return float(s.iloc[k]) if k >= 0 else np.nan

    rows = []
    for ln in lines:
        ee = [x for x in ent[ln]]
        for k, (m, sd, nd, pk) in enumerate(ee):
            nxt = ee[k + 1][1] if k + 1 < len(ee) else None
            end = nxt if nxt is not None else asof
            st = 'complete' if nxt is not None else ('pending_exec' if nd is None else 'open')
            r50 = at(E, end) / at(E, sd) - 1
            row = dict(line='MAIN_OFFICIAL' if ln == 'official' else ln.upper(), anchor=str(m), start=str(sd.date()), end=str(end.date()), status=st,
                       ret_0050=r50, cost='V7X K5T GROW 100萬（一般）', n_hold=len(pk) if pk else 0, source='make_forward v3')
            if nd is not None and (ln, 'normal') in navs:
                for var, rc, xc in (('normal', 'ret', 'excess_pp'), ('stress', 'ret_stress', 'excess_stress_pp'), ('cap10', 'ret_cap10', 'excess_cap10_pp')):
                    nv = navs[(ln, var)]; r = at(nv, end) / at(nv, sd) - 1
                    row[rc] = r; row[xc] = (r - r50) * 100
            rows.append(row)
    fr = pd.DataFrame(rows, columns=FR_COLS)
    jm = []
    J = fr[fr.line.str.match(r'^J\d+$')]
    for m, g in J.groupby('anchor'):
        comp = g[g.status == 'complete']
        st = 'complete' if len(comp) == NJ else 'open'
        src_ = comp if st == 'complete' else g[g.excess_pp.notna()]
        if not len(src_): continue
        jm.append(dict(line='JMED', anchor=m, start=g.start.min(), end=g.end.max(), status=st, n_j=len(comp),
                       excess_pp=float(src_.excess_pp.median()), excess_stress_pp=float(src_.excess_stress_pp.median()),
                       excess_cap10_pp=float(src_.excess_cap10_pp.median()), cost='V7X K5T（13 個 J 窗超額的中位數）', source='make_forward v3',
                       note='' if st == 'complete' else f'只有 {len(comp)}/13 個窗完成；中位數用目前可得的窗，未定稿'))
    off = fr[(fr.line == 'MAIN_OFFICIAL')].to_dict('records')
    e50rows = [dict(r, line='E50', ret=r['ret_0050'], excess_pp=0.0, ret_stress=np.nan, excess_stress_pp=np.nan, ret_cap10=np.nan,
                    excess_cap10_pp=np.nan, cost='0050 含息（finlab adj）', n_hold=1) for r in off]
    # ---- 4. 處置衛星 ----
    dstart = pd.Timestamp(a.disp_start) if a.disp_start else cal[cal >= start.to_timestamp()][0] if (cal >= start.to_timestamp()).any() else asof
    frd = DS.load_frames(V.dir)
    FD.assert_asof(asof, disp_C=frd['C'], disp_dsp=frd['dsp'], taiex=frd['tx'], disp_e50=frd['e50'])
    V.drop()
    ev = DS.events(frd)
    led, n_new = DS.update_ledger(ev, out / 'disposal_ledger.csv', clock, ckind, asof)
    dsim = DS.simulate(frd, ev, led, dstart) if dstart <= asof else {}
    dsim_s = DS.simulate(frd, ev, led, dstart, stress=True) if dstart <= asof else {}
    dnav = {}
    drows = []
    dec_rows = []
    for ln, (nv, fl, dec) in dsim.items():
        dnav[ln] = nv
        if len(fl): fills += [dict(r, kind='paper_disp', is_odd=int(r['shares'] % 1000 != 0)) for r in fl.to_dict('records')]
        for eid, d in dec.items(): dec_rows.append(dict(line=ln, event_id=eid, decision=d))
        nvs = dsim_s[ln][0]
        for r in off:
            sd, end = pd.Timestamp(r['start']), pd.Timestamp(r['end'])
            if len(nv) == 0 or sd < nv.index[0]: continue
            rr = at(nv, end) / at(nv, sd) - 1; rs = at(nvs, end) / at(nvs, sd) - 1
            drows.append(dict(line=ln, anchor=r['anchor'], start=r['start'], end=r['end'], status=r['status'], ret=rr, ret_0050=r['ret_0050'],
                              excess_pp=(rr - r['ret_0050']) * 100, ret_stress=rs, excess_stress_pp=(rs - r['ret_0050']) * 100,
                              cost='處置可執行層 v1_cap20（收盤±跳＋0.5σ√(q/Qref)）、閒置 0050', source='disp_sat.py', note=f'起算 {dstart.date()}'))
    if dnav:
        DN = pd.DataFrame({k: v for k, v in dnav.items()})
        DN.index.name = 'date'
        FD.write_csv(DN.reset_index().assign(date=lambda x: x['date'].dt.strftime('%Y-%m-%d')), out / 'disp_nav.csv', gz=False)
    FD.write_csv(pd.DataFrame(dec_rows, columns=['line', 'event_id', 'decision']), out / 'disposal_decisions.csv', gz=False)
    # ---- 5. 混合（窗初再平衡；漂移流量 × 0.5% 再平衡成本） ----
    def mix(base_rows, w, line, label):
        res = []; wd = None
        for r in base_rows:
            if not np.isfinite(r.get('ret', np.nan)): continue
            cost = abs(wd - w) * MIX_REB_COST if wd is not None else 0.0
            rm = w * r['ret'] + (1 - w) * r['ret_0050'] - cost
            rs = w * r['ret_stress'] + (1 - w) * r['ret_0050'] - cost if np.isfinite(r.get('ret_stress', np.nan)) else np.nan
            res.append(dict(line=line, anchor=r['anchor'], start=r['start'], end=r['end'], status=r['status'], ret=rm, ret_0050=r['ret_0050'],
                            excess_pp=(rm - r['ret_0050']) * 100, ret_stress=rs, excess_stress_pp=(rs - r['ret_0050']) * 100, cost=label, source='make_forward v3'))
            wd = w * (1 + r['ret']) / (w * (1 + r['ret']) + (1 - w) * (1 + r['ret_0050']))
        return res
    mixes = mix(off, 0.30, 'MIX_C70', '0050 70%＋主策略官方日 30%（窗初再平衡，漂移流量×0.5%）') + \
        mix([r for r in drows if r['line'] == 'DISP_TWSE'], 0.20, 'MIX_D80', '0050 80%＋處置衛星（上市）20%（窗初再平衡，漂移流量×0.5%）')
    FR = pd.concat([fr, pd.DataFrame(jm), pd.DataFrame(e50rows), pd.DataFrame(drows), pd.DataFrame(mixes)], ignore_index=True).reindex(columns=FR_COLS)
    FR = FR.sort_values(['line', 'anchor', 'start'], na_position='last').reset_index(drop=True)
    FD.write_csv(FR, out / 'forward_returns.csv', gz=False)
    FD.write_csv(pd.DataFrame([dict(start=r['start'], end=r['end'], ret=r.get('ret')) for r in off], columns=['start', 'end', 'ret']),
                 out / 'main_leg_returns.csv', gz=False)
    # ---- 6. trades.csv（紙上＋實際） ----
    fin = out / 'fills_input.csv'
    if not fin.exists(): FD.write_csv(pd.DataFrame(columns=FILL_FIELDS), fin, gz=False)
    F_ = pd.read_csv(fin, dtype={'code': str})
    for _, r in F_.iterrows():
        d = pd.Timestamp(r['date']); c = r['code']
        if d not in dmod.TPOS or c not in dmod.CI: continue
        t, j = dmod.TPOS[d], dmod.CI[c]
        o = dmod.O[t, j]; tk = float(XL.tick(o)); sgn = 1.0 if str(r['side']).lower().startswith('b') else -1.0
        v60 = dmod.V60p[t, j]; oa = XE.OASH[t, j] * v60; q = float(r['shares']) * dmod.PC[t, j]
        oaa = XE.OASH[t, j] * dmod.V[t, j]
        fills.append(dict(kind='real', line=r.get('list', ''), date=str(d.date()), code=c, side=r['side'], is_odd=r['is_odd'], shares=r['shares'],
                          fill_px=r['fill_px'], open_px=o, prev_close=dmod.PC[t, j], close_px=dmod.C[t, j], ticks_from_open=sgn * (float(r['fill_px']) - o) / tk,
                          sigma60_prev=dmod.SIGp[t, j], V60_prev=v60, dec=int(XE.DEC[t, j]), oa_share_dec=XE.OASH[t, j], OA_pred=oa,
                          share_of_open_vol=q / oa if oa > 0 else np.nan, share_of_open_vol_actual=q / oaa if oaa > 0 else np.nan,
                          in_disposal=int(bool(dmod.DISP[t, j])), order_time=r.get('order_time', ''), limit_px=r.get('limit_px', np.nan)))
    TRD = pd.DataFrame(fills).reindex(columns=TR_COLS)
    if len(TRD): TRD = TRD.sort_values(['kind', 'line', 'date', 'code', 'side', 'is_odd'], kind='stable').reset_index(drop=True)
    FD.write_csv(TRD, out / 'trades.csv', gz=False)
    # ---- 7. G8 追蹤（只記錄；判定在 12 個錨定月完成後） ----
    dl = []
    g8_from = str(pd.Period(prereg['G8']['first_anchor'], 'M')) if prereg else str(start)
    for ln in G8_LINES + ['MAIN_OFFICIAL']:
        g = FR[(FR.line == ln) & (FR.status == 'complete') & (FR.anchor.astype(str) >= g8_from)].sort_values('anchor').head(12)
        x = g.excess_pp.to_numpy(float)
        dl.append(dict(line=ln, n_complete=len(x), cum_excess_pp=float(x.sum()) if len(x) else 0.0,
                       mean_excess_pp=float(x.mean()) if len(x) else np.nan,
                       state='ACCUMULATING' if len(x) < 12 else ('PASS' if x.sum() > 0 else 'FAIL'),
                       g8_line=int(ln in G8_LINES)))
    FD.write_csv(pd.DataFrame(dl), out / 'decision_log.csv', gz=False)
    # ---- 8. list_diffs、runs ----
    if diffs:
        f = out / 'list_diffs.csv'
        old = pd.read_csv(f, dtype=str) if f.exists() else pd.DataFrame()
        FD.write_csv(pd.concat([old, pd.DataFrame(diffs).astype(str)]).drop_duplicates(), f, gz=False)
    outs = {p.name: sha(p) for p in sorted(out.glob('*.csv'))}
    outs.update({f'paper_lists/{p.name}': sha(p) for p in sorted(pl_dir.glob('*'))})
    rec.update(status='OK', months=[str(m) for m in months], frozen_now=frozen_now, n_list_diffs=len(diffs), n_fills=len(TRD),
               n_ledger_new=n_new, disp_start=str(dstart.date()), outputs=outs)
    with open(out / 'runs.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('outputs', 'dir_check', 'loaded_max_dates')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
