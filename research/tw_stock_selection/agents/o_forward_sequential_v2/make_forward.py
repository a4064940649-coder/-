#!/usr/bin/env python3
"""2026-11 起前瞻追蹤產生器（o_forward_sequential_v2；PREREG_FORWARD 登記本檔 sha256）。可重複執行、冪等。

用法（每月換股日收盤後；資料先 git pull）：
  python3 -I -B make_forward.py --data <tw-stock-screener>/data --e50 <finlab-txd>/data/raw/etf_0050.csv \
      [--out agents/live_tracking/main_forward] [--start 2026-11] [--asof YYYY-MM-DD]
乾跑（重現 2026-10-12 SOP 名單、2026-09／10 對 V7 對帳）：s6_dryrun.py 以 --start 2026-09 --out forward_dry 呼叫本檔。

定義（與 V7 j 掃描一致，o_v7_listing_exec_engine/s4_jscan7.py）
- 錨定月 m 的法定期限 D_m＝m 月 10 日起第一個交易日；J{j}＝D_m 之後第 j 個交易日（J0＝D_m）。官方日＝m 月第一個 ≥ 11 日的交易日
  （10 日為交易日時官方日＝J1）。
- 名單（全部排除金融保險、處置中、KY 等同 main_picks_today.py --ex-fin）：
  官方日：直接呼叫 main_picks_today.score_on／execute（＝SOP 工具），價格資料模式與該工具在訊號日執行時相同。
  J 線：同一分數，但營收只用「訊號日收盤前已公告」者：m−1 月營收的 ann_date ≤ 訊號日（ann_date 缺值視為期限日 D_m 已公告），
  否則該檔沿用 m−2..m−4 的 3 月年增率（上月分數）。
- 成交（M2）：訊號日隔一個交易日開盤；開盤漲停而且沒持有的不買，由下一名遞補；停牌不買。續抱股調回等權。
- 紙上報酬：逐日引擎（收盤對收盤；換股日＝前收→開盤用舊持股、開盤換股付成本、開盤→收盤用新持股）。
  成本（K5T 小帳戶近似，一般）：每邊 手續費 0.1425%×0.28 ＋ max(0.1%, 一跳/開盤)；賣出另加證交稅 0.3%。壓力：每邊再加 0.4%、手續費 6 折。
  窗＝該線訊號日收盤 → 下一個訊號日收盤（與回測 t 的窗相同）；0050＝finlab 含息 adj 同窗。
- JMED（決策線）：錨定月 m 的 13 個 J 線窗超額取中位數；13 個窗都完成才定稿（status=complete）。
- 名單快照：每個 (線, 月) 的名單第一次寫入後凍結；之後重算若不同，寫 list_diffs.csv，不覆寫。
產出（--out）：forward_returns.csv、trades.csv（由 fills_input.csv 補欄）、paper_lists/{j0,j1,j6,jall,official}_YYYYMM.csv、
  main_leg_returns.csv（給 o7_regime）、decision_log.csv（PREREG_FORWARD 凍結後）、list_diffs.csv、runs.jsonl。
本檔只寫 --out 目錄；其他資料夾只讀。
"""
import argparse, json, hashlib, sys, math, datetime, subprocess
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
TW = AG.parent
sys.path.insert(0, str(TW))
import main_picks_today as MP   # noqa: E402（只讀 import）
sys.path.pop(0)
sys.path.insert(0, str(HERE))
import seq2 as Q                # noqa: E402
sys.path.pop(0)

FEE_N, FEE_S, TAX, BASE, STRESS_X = 0.001425 * 0.28, 0.001425 * 0.6, 0.003, 0.001, 0.004
NJ = 13
TRADE_FIELDS = ['date', 'code', 'side', 'order_time', 'limit_px', 'fill_px', 'open_px', 'prev_close', 'ticks_from_open',
                'shares', 'is_odd', 'sigma60_prev', 'V60_prev', 'oa_share_dec', 'share_of_open_vol', 'list']
FILL_FIELDS = ['date', 'code', 'side', 'order_time', 'limit_px', 'fill_px', 'shares', 'is_odd', 'list']
EXTERNAL = {
    'CAPM': AG / 'live_tracking/o7_capmatched/paper_nav.tsv',
    'REGIME_LEDGER': AG / 'live_tracking/o7_regime/ledger.csv',
    'REGIME_REG': AG / 'live_tracking/o7_regime/registration.json',
    'RROLL': AG / 'o_rev_freshness_attention/forward/rroll_trades.csv',
    'R2': AG / 'o_rev_freshness_attention/forward/r2_events.csv',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 資料
class Store:
    """MP.load_prices 的快取：依訊號日決定模式（與 main_picks_today 在該日執行時相同）。"""
    def __init__(self, D, panel):
        self.D, self.panel = Path(D), Path(panel)
        self.hcal = pd.DatetimeIndex(sorted(pd.read_csv(self.D / 'history.csv.gz', usecols=['date'], parse_dates=['date'])['date'].unique()))
        self.cache = {}

    def mode_for(self, sd):
        k = self.hcal.get_loc(sd) if sd in self.hcal else -1
        return 'hist' if k >= MP.W52 - 1 else 'stitch'

    def get(self, mode):
        if mode not in self.cache:
            tgt = self.hcal[-1] if mode == 'hist' else self.hcal[min(MP.W52 - 10, len(self.hcal) - 1)]
            self.cache[mode] = MP.load_prices(self.D, tgt, self.panel)
        return self.cache[mode]


def calendar(cal):
    """錨定月 → (D_m, [J0..J12], 官方日)。"""
    out = {}
    for p in pd.period_range(pd.Period(cal[0], 'M'), pd.Period(cal[-1], 'M'), freq='M'):
        c = cal[(cal.year == p.year) & (cal.month == p.month) & (cal.day >= 10)]
        if not len(c): continue
        k = cal.get_loc(c[0])
        js = [cal[k + j] if k + j < len(cal) else None for j in range(NJ)]
        off = MP.rebalance_day(cal, p.year, p.month)
        out[p] = dict(D=c[0], J=js, official=off)
    return out


def ann_table(D):
    r = pd.read_csv(Path(D) / 'extras/rev_hist.csv.gz', dtype={'code': str})
    r = r[r['ann_date'].notna()]
    return {(c, y): pd.Timestamp(a) for c, y, a in zip(r.code, r.ym, r.ann_date)}


def is_fin(data, cols):
    return pd.Series(cols.map(lambda c: data['ind'].get(c, '') == MP.FIN_IND or c[:2] == '28' or c in ('5876', '5880')), index=cols)


def score_j(data, YOY3, dsp, sd, anchor, ann, Dm):
    """J 線分數：同 MP.score_on，但 m−1 月營收只在已公告時使用。回傳 (order, full)。"""
    _, df, _, _ = MP.score_on(data, YOY3, dsp, sd, ex_fin=True)
    cols = df.index
    pm1, pm2 = anchor - 1, anchor - 2
    new = YOY3.loc[pm1].reindex(cols) if pm1 in YOY3.index else pd.Series(np.nan, index=cols)
    old = YOY3.loc[pm2].reindex(cols) if pm2 in YOY3.index else pd.Series(np.nan, index=cols)
    ok = pd.Series([(ann.get((c, str(pm1)), Dm) <= sd) for c in cols], index=cols)
    y3 = new.where(ok, old)
    el = df['t300'] & y3.notna() & df['s52'].notna() & ~df['inD'] & ~is_fin(data, cols)
    sc = 0.5 * df['s52'].where(el).rank(pct=True) + 0.5 * y3.where(el).rank(pct=True)
    full = df.assign(yoy3=y3, el=el, score=sc, rev_month=np.where(ok, str(pm1), str(pm2)))
    full['score_r'] = full['score'].round(10)
    order = full[full['el']].reset_index().sort_values(['score_r', 'yoy3', 'code'], ascending=[False, False, True]).drop(columns='score_r').set_index('code')
    return order, full


def score_official(data, YOY3, dsp, sd):
    order, full, m, n_el = MP.score_on(data, YOY3, dsp, sd, ex_fin=True)
    full = full.assign(rev_month=str(m))
    return order, full


# ---------------------------------------------------------------- 逐日紙上引擎
def cost_side(o, side, stress):
    t = MP.tick(np.asarray(o, float))
    imp = np.maximum(BASE, t / np.asarray(o, float))
    fee = FEE_S if stress else FEE_N
    c = fee + imp + (TAX if side == 'sell' else 0.0)
    return c + (STRESS_X if stress else 0.0)


def run_line(entries, px, start, end, stress=False):
    """entries：[(signal_day, exec_day, [codes])]（目標持股，等權）。px：dict(R, O, C, ADJ, cal)。
    回傳日淨值（start 收盤＝1）與每日持股數。"""
    cal = px['cal']; days = cal[(cal > start) & (cal <= end)]
    R, O, ADJ = px['R'], px['O'], px['ADJ']
    ex = {e: (s, c) for s, e, c in entries}
    w = pd.Series(dtype=float); nav = 1.0; out = {start: 1.0}; nh = {}
    prev = start
    for t in days:
        if t in ex:
            codes = ex[t][1]
            if len(w):
                ao = (O.loc[t].reindex(w.index) * ADJ.loc[t].reindex(w.index) / px['C'].loc[t].reindex(w.index))
                ac0 = ADJ.loc[prev].reindex(w.index)
                g = (ao / ac0).where(np.isfinite(ao / ac0), 1.0)
                nav *= float((w * g).sum()); w = w * g; w = w / w.sum()
            # 停牌而且要賣的持股：保留
            o_t = O.loc[t]
            stuck = [c for c in w.index if c not in codes and not np.isfinite(o_t.get(c, np.nan))]
            keepw = float(w.reindex(stuck).sum()) if stuck else 0.0
            tgt = pd.Series((1 - keepw) / max(len(codes), 1), index=list(codes))
            if stuck: tgt = pd.concat([tgt, w.reindex(stuck)])
            allc = tgt.index.union(w.index)
            dlt = tgt.reindex(allc).fillna(0) - w.reindex(allc).fillna(0)
            oo = o_t.reindex(allc)
            cb = cost_side(oo.fillna(1e9).to_numpy(), 'buy', stress); cs = cost_side(oo.fillna(1e9).to_numpy(), 'sell', stress)
            cost = float((np.where(dlt > 0, dlt * cb, -dlt * cs)).sum())
            nav *= (1 - cost)
            w = tgt[tgt > 0]
            ao = (O.loc[t].reindex(w.index) * ADJ.loc[t].reindex(w.index) / px['C'].loc[t].reindex(w.index))
            g = (ADJ.loc[t].reindex(w.index) / ao)
            g = g.where(np.isfinite(g), (1 + R.loc[t].reindex(w.index).fillna(0)))
            nav *= float((w * g).sum()); w = w * g; w = w / w.sum()
        elif len(w):
            g = 1 + R.loc[t].reindex(w.index).fillna(0.0)
            nav *= float((w * g).sum()); w = w * g; w = w / w.sum()
        out[t] = nav; nh[t] = len(w); prev = t
    return pd.Series(out), pd.Series(nh)


def at(s, d):
    s = s.dropna(); k = s.index.searchsorted(d, side='right') - 1
    return float(s.iloc[k]) if k >= 0 else np.nan


# ---------------------------------------------------------------- 主流程
def build_lists(store, C, months, asof, ann, YOY3, dsp, out_dir, diffs):
    """回傳 dict(line → [(anchor, signal, exec, picks, table)])；名單快照凍結。"""
    pl_dir = out_dir / 'paper_lists'; pl_dir.mkdir(parents=True, exist_ok=True)
    lists = {f'J{j}': [] for j in range(NJ)}; lists['OFFICIAL'] = []
    held = {k: set() for k in lists}
    for m in months:
        cm = C.get(m)
        if cm is None: continue
        days = [('OFFICIAL', cm['official'])] + [(f'J{j}', cm['J'][j]) for j in range(NJ)]
        for line, sd in days:
            if sd is None or sd > asof: continue
            data = store.get(store.mode_for(sd))
            cal = data['cal']
            if sd not in cal: continue
            if line == 'OFFICIAL':
                order, full = score_official(data, YOY3, dsp, sd)
            else:
                order, full = score_j(data, YOY3, dsp, sd, m, ann, cm['D'])
            ex = MP.execute(data, order, sd, 10, held[line])
            if ex is None:
                continue                                   # 隔日資料還沒有：下次再算
            picks, nd = ex['picks'], ex['next_day']
            tab = make_table(data, order, full, picks, ex, sd, nd, dsp, line, m)
            snap = pl_dir / f'{line.lower()}_{m.strftime("%Y%m")}.csv'
            if snap.exists():
                old = pd.read_csv(snap, dtype={'code': str})
                oldp = list(old[old.bought == 1].sort_values('rank').code)
                if oldp != picks:
                    diffs.append(dict(line=line, anchor=str(m), signal=str(sd.date()), snapshot=','.join(oldp), recomputed=','.join(picks)))
                picks = oldp
            else:
                tab.to_csv(snap, index=False)
            lists[line].append((m, sd, nd, picks))
            held[line] = set(picks)
    # 規定檔名：j0／j1／j6（已由上面寫成 j0_、j1_、j6_）；全部 J 合併檔
    for m in months:
        fs = [pl_dir / f'j{j}_{m.strftime("%Y%m")}.csv' for j in range(NJ)]
        if all(f.exists() for f in fs):
            allf = pl_dir / f'jall_{m.strftime("%Y%m")}.csv'
            if not allf.exists():
                pd.concat([pd.read_csv(f, dtype={'code': str}) for f in fs]).to_csv(allf, index=False)
    return lists


def make_table(data, order, full, picks, ex, sd, nd, dsp, line, m):
    O, Cc = data['O'], data['C']
    cand = list(order.index[:max(20, order.index.get_loc(picks[-1]) + 1 if picks else 20)])
    i = data['cal'].get_loc(sd)
    prevc = Cc.iloc[:i + 1].ffill().iloc[-1]
    rows = []
    sk = dict(ex['skipped'])
    for code in cand:
        r = full.loc[code]
        o = O.loc[nd].get(code, np.nan); c = Cc.loc[nd].get(code, np.nan)
        r1 = data['R'].loc[nd].get(code, np.nan)
        ref = c / (1 + r1) if np.isfinite(r1) and np.isfinite(c) else prevc.get(code, np.nan)
        lu = float(MP.limit_up_price(np.array([ref]))[0]) if np.isfinite(ref) else np.nan
        d_sig = dsp[(dsp.code == code) & (dsp.start <= sd) & (dsp.end >= sd)]
        d_ann = dsp[(dsp.code == code) & (dsp.date <= sd) & (dsp.start > sd)]
        d_ex = dsp[(dsp.code == code) & (dsp.start <= nd) & (dsp.end >= nd)]
        rows.append(dict(line=line, anchor=str(m), signal=str(sd.date()), exec=str(nd.date()), rank=int(order.index.get_loc(code)) + 1, code=code,
                         name=data['names'].get(code, ''), score=float(r['score']), s52=float(r['s52']), yoy3=float(r['yoy3']), rev_month=r.get('rev_month', ''),
                         bought=int(code in picks), skipped=sk.get(code, ''), open=o, close=c, limit_up_px=lu,
                         open_limit_up=int(np.isfinite(o) and np.isfinite(lu) and o >= lu - 1e-6),
                         close_limit_up=int(np.isfinite(c) and np.isfinite(lu) and c >= lu - 1e-6),
                         disp_on_signal=int(len(d_sig) > 0), disp_announced_by_signal=int(len(d_ann) > 0), disp_on_exec=int(len(d_ex) > 0)))
    return pd.DataFrame(rows)


def windows_for(entries, nav, E, end_cap, line, stress_nav=None):
    rows = []
    for k, (m, sd, nd, picks) in enumerate(entries):
        nxt = entries[k + 1][1] if k + 1 < len(entries) else None
        b = nxt if nxt is not None else min(nav.index[-1], end_cap)
        st = 'complete' if nxt is not None else 'open'
        r = at(nav, b) / at(nav, sd) - 1; r50 = at(E, b) / at(E, sd) - 1
        row = dict(line=line, anchor=str(m), start=str(sd.date()), end=str(b.date()), status=st, ret=r, ret_0050=r50,
                   excess_pp=(r - r50) * 100, cost='K5T-lite normal', n_hold=len(picks), source='make_forward')
        if stress_nav is not None:
            rs = at(stress_nav, b) / at(stress_nav, sd) - 1
            row['ret_stress'] = rs; row['excess_stress_pp'] = (rs - r50) * 100
        rows.append(row)
    return rows


def mix_rows(base, other, w_other, line, E, sd_end, label):
    rows = []
    for r in base:
        a, b = pd.Timestamp(r['start']), pd.Timestamp(r['end'])
        ro = other(a, b)
        if ro is None or not np.isfinite(ro): continue
        rm = (1 - w_other) * r['ret'] + w_other * ro
        rows.append(dict(r, line=line, ret=rm, excess_pp=(rm - r['ret_0050']) * 100, cost=label, ret_stress=np.nan, excess_stress_pp=np.nan))
    return rows


def external_rows(off_rows, E, px):
    rows = []
    # 市值匹配 F／C50／C70
    f = EXTERNAL['CAPM']
    if f.exists():
        P_ = pd.read_csv(f, sep='\t', parse_dates=['exec', 'next_exec'])
        for _, r in P_.iterrows():
            for ln, col in (('CAPM_F1', 'r_F'), ('CAPM_C50', 'r_C50'), ('CAPM_C70', 'r_C70')):
                rows.append(dict(line=ln, anchor=str(pd.Period(r['exec'], 'M')), start=str(r['exec'].date()), end=str(r['next_exec'].date()), status='complete',
                                 ret=r[col], ret_0050=r['r_0050'], excess_pp=(r[col] - r['r_0050']) * 100, cost='o7_paper.py', source=str(f)))
    else:
        for ln in ('CAPM_F1', 'CAPM_C50', 'CAPM_C70'):
            rows.append(dict(line=ln, status='pending_external', source=str(f), note='o7_paper.py 尚未產生 paper_nav.tsv'))
    # 輪動 V5：ledger 的 w0050 × 0050 ＋ (1−w) × 主策略官方日
    f = EXTERNAL['REGIME_LEDGER']
    if f.exists() and EXTERNAL['REGIME_REG'].exists():
        led = pd.read_csv(f, parse_dates=['decision'])
        reg = json.loads(EXTERNAL['REGIME_REG'].read_text(encoding='utf-8')); c = reg.get('costs_normal', [0, 0, 0, 0])
        wprev = None
        for r in off_rows:
            l = led[led.decision == pd.Timestamp(r['start'])]
            if not len(l): continue
            w = float(l.iloc[0]['w0050'])
            sw = 0.0 if wprev is None else (abs(w - wprev) * ((c[1] + c[2]) if w > wprev else (c[3] + c[0])))
            rr = w * r['ret_0050'] + (1 - w) * r['ret'] - sw
            rows.append(dict(r, line='REGIME_V5', ret=rr, excess_pp=(rr - r['ret_0050']) * 100, cost='regime costs_normal', source=str(f), ret_stress=np.nan, excess_stress_pp=np.nan))
            wprev = w
    else:
        rows.append(dict(line='REGIME_V5', status='pending_external', source=str(f), note='o7_regime_live.py 尚未產生 ledger.csv'))
    # R_roll：依交易紀錄重建持股，隔日開盤成交，等權
    f = EXTERNAL['RROLL']
    if f.exists() and len(off_rows):
        T_ = pd.read_csv(f, dtype={'code': str}, parse_dates=['signal'])
        cal = px['cal']; H_ = set(); ent = []
        for sd, g in T_.groupby('signal'):
            H_ = (H_ - set(g[g.side == 'sell'].code)) | set(g[g.side == 'buy'].code)
            k = cal.searchsorted(sd, side='right')
            if k < len(cal): ent.append((sd, cal[k], sorted(H_)))
        if ent:
            st = ent[0][0]; nav, _ = run_line(ent, px, st, cal[-1])
            for r in off_rows:
                a, b = pd.Timestamp(r['start']), pd.Timestamp(r['end'])
                if a < st: continue
                rr = at(nav, b) / at(nav, a) - 1
                rows.append(dict(r, line='R_ROLL', ret=rr, excess_pp=(rr - r['ret_0050']) * 100, cost='K5T-lite normal（等權近似）', source=str(f), ret_stress=np.nan, excess_stress_pp=np.nan))
    else:
        rows.append(dict(line='R_ROLL', status='pending_external', source=str(f), note='forward_track.py 尚未產生 rroll_trades.csv'))
    rows.append(dict(line='R2', status='pending_external', source=str(EXTERNAL['R2']),
                     note='R2 候選已判不可用；X 的權重定義須由 o_rev_freshness_attention 提供，本檔只記事件檔是否存在：' + str(EXTERNAL['R2'].exists())))
    for ln in ('T1', 'T2'):
        rows.append(dict(line=ln, status='pending_external', source='agents/o_0050_tilt_pit',
                         note='題 2 判不可用，未登記前瞻腳本；有 forward 輸出後再接'))
    return rows


def enrich_trades(out_dir, px, spec):
    fin = out_dir / 'fills_input.csv'
    if not fin.exists():
        pd.DataFrame(columns=FILL_FIELDS).to_csv(fin, index=False)
    F_ = pd.read_csv(fin, dtype={'code': str})
    rows = []
    cal = px['cal']; O, C, V, R = px['O'], px['C'], px['V'], px['R']
    good = cal[(C.notna().sum(axis=1) >= MP.MIN_NOBS).to_numpy()]
    oa = spec['oa_share_by_dec']
    for _, r in F_.iterrows():
        d = pd.Timestamp(r['date']); c = r['code']
        if d not in cal: continue
        k = good.searchsorted(d) - 1
        win = good[max(k - 59, 0):k + 1]
        v60 = V.loc[win, c].mean() if c in V.columns else np.nan
        s60 = R.loc[win, c].std() if c in R.columns else np.nan
        vv = V.loc[win].mean(); px_ = C.loc[win[-1]]
        el = (vv >= MP.MIN_VAL) & (px_ >= MP.MIN_PX)
        vr = vv.where(el).rank(ascending=False).get(c, np.nan)
        dec = int(min(10, math.ceil(vr / 30))) if np.isfinite(vr) and vr <= 300 else 10
        o = O.loc[d].get(c, np.nan); pc = C.loc[:d].iloc[:-1][c].dropna().iloc[-1] if c in C.columns else np.nan
        sgn = 1.0 if str(r['side']).lower().startswith('b') else -1.0
        tk = float(MP.tick(np.array([o]))[0]) if np.isfinite(o) else np.nan
        q = float(r['shares']) * pc if np.isfinite(pc) else np.nan
        oap = oa[str(dec)] * v60 if np.isfinite(v60) else np.nan
        rows.append(dict(date=str(d.date()), code=c, side=r['side'], order_time=r.get('order_time', ''), limit_px=r.get('limit_px', np.nan), fill_px=r['fill_px'],
                         open_px=o, prev_close=pc, ticks_from_open=sgn * (float(r['fill_px']) - o) / tk if np.isfinite(tk) else np.nan,
                         shares=r['shares'], is_odd=r['is_odd'], sigma60_prev=s60, V60_prev=v60, oa_share_dec=oa[str(dec)],
                         share_of_open_vol=q / oap if oap and np.isfinite(oap) else np.nan, list=r.get('list', '')))
    pd.DataFrame(rows, columns=TRADE_FIELDS).to_csv(out_dir / 'trades.csv', index=False)
    return len(rows)


def decision_log(out_dir, fr, rule_f):
    """PREREG_FORWARD 登記後才寫。rule.name 為 None（OC 未過、規則未凍結）時只記描述統計（z、後驗機率），state＝DESCRIPTIVE_ONLY。"""
    if not rule_f.exists(): return 'no_rule'
    res = json.loads(rule_f.read_text(encoding='utf-8'))
    if 'PREREG_FORWARD' not in res: return 'no_rule'
    R = res['PREREG_FORWARD']['rule']
    z = np.load(HERE / R['prior_file'])
    th, prior = z['theta'], z['prior']
    d = fr[(fr.line == 'JMED') & (fr.status == 'complete')].sort_values('anchor')
    d = d[d.anchor.astype(str) >= R['forward_start']]
    rows = []
    x = d.excess_pp.to_numpy()
    sigma = R['sigma_design']
    par = dict(R['params'])
    for n in range(1, len(x) + 1):
        X = x[None, :n]
        sd_fwd = float(np.std(x[:n], ddof=1)) if n >= 2 else np.nan
        sig = sigma
        if n >= 12 and sd_fwd > 1.5 * sigma: sig = sd_fwd           # 唯一允許的 refit（升級邊界只能變嚴：見 PREREG_FORWARD）
        pp = Q.post_probs(X, sig, prior, th, [n], par.get('alpha_min', 0.25))[n]
        row = dict(month=d.anchor.iloc[n - 1], n=n, S=float(x[:n].sum()), z=float(x[:n].sum() / (sig * math.sqrt(n))), sigma_used=sig, sd_forward=sd_fwd,
                   P_alpha_gt_min=float(pp[0][0]), P_alpha_lt0=float(pp[1][0]))
        if R.get('name'):
            up, dn = Q.raw_triggers(R['name'], X, sig, par, prior, th)
            u, dd = Q.first_events(up, dn)
            row.update(upgrade=int(u[0] > 0), downgrade=int(dd[0] > 0), state='UPGRADE' if u[0] > 0 else ('DOWNGRADE' if dd[0] > 0 else 'CONTINUE'))
        else:
            row.update(upgrade=0, downgrade=0, state='DESCRIPTIVE_ONLY')
        rows.append(row)
    pd.DataFrame(rows, columns=['month', 'n', 'S', 'z', 'sigma_used', 'sd_forward', 'P_alpha_gt_min', 'P_alpha_lt0', 'upgrade', 'downgrade', 'state']).to_csv(out_dir / 'decision_log.csv', index=False)
    return f'{len(rows)} months' + ('' if R.get('name') else '（規則未凍結，只記描述）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(HERE / 'raw/screener/data'))
    ap.add_argument('--e50', default=str(HERE / 'raw/finlab/data/raw/etf_0050.csv'))
    ap.add_argument('--panel', default=str(MP.PANEL))
    ap.add_argument('--out', default=str(AG / 'live_tracking/main_forward'))
    ap.add_argument('--start', default='2026-11')
    ap.add_argument('--asof')
    ap.add_argument('--rule', default=str(HERE / 'results.json'))
    ap.add_argument('--spec', default=str(HERE / 'raw/v7/model_spec7.json'))
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(a.data, a.panel)
    asof = pd.Timestamp(a.asof) if a.asof else store.hcal[-1]
    px = store.get('hist')
    cal = px['cal'][px['cal'] <= asof]
    C = calendar(px['cal'])
    start = pd.Period(a.start, 'M')
    months = [m for m in sorted(C) if m >= start and C[m]['D'] <= asof]
    _, YOY3, _ = MP.load_revenue(Path(a.data))
    dsp = MP.load_disposal(Path(a.data))
    ann = ann_table(a.data)
    diffs = []
    lists = build_lists(store, C, months, asof, ann, YOY3, dsp, out_dir, diffs)
    E = pd.read_csv(a.e50, parse_dates=['date']).set_index('date')['adj'].sort_index()
    end_cap = min(asof, E.index[-1])
    pxr = dict(px); pxr['cal'] = cal
    rows = []
    navs = {}
    for line, ent in lists.items():
        if not ent: continue
        st = ent[0][1]
        e2 = [(s, e, p) for (_, s, e, p) in ent]
        nav, _ = run_line(e2, pxr, st, end_cap)
        nvs, _ = run_line(e2, pxr, st, end_cap, stress=True)
        navs[line] = nav
        nm = 'MAIN_OFFICIAL' if line == 'OFFICIAL' else line
        rows += windows_for(ent, nav, E, end_cap, nm, nvs)
    fr = pd.DataFrame(rows)
    # JMED
    jm = []
    if len(fr):
        J = fr[fr.line.str.match(r'^J\d+$')]
        for m, g in J.groupby('anchor'):
            comp = g[g.status == 'complete']
            st = 'complete' if len(comp) == NJ else 'open'
            src = comp if st == 'complete' else g
            jm.append(dict(line='JMED', anchor=m, start=src.start.min(), end=src.end.max(), status=st, n_j=len(comp),
                           excess_pp=float(src.excess_pp.median()), excess_stress_pp=float(src.excess_stress_pp.median()),
                           ret=np.nan, ret_0050=np.nan, cost='K5T-lite normal（13 j 中位數）', source='make_forward'))
    off = [r for r in rows if r['line'] == 'MAIN_OFFICIAL']
    E50 = [dict(r, line='E50', ret=r['ret_0050'], excess_pp=0.0, cost='0050 含息', ret_stress=np.nan, excess_stress_pp=np.nan) for r in off]
    s2330 = px['ADJ']['2330'] if '2330' in px['ADJ'].columns else None
    f2330 = (lambda a_, b_: at(s2330, b_) / at(s2330, a_) - 1) if s2330 is not None else (lambda a_, b_: None)
    fE = lambda a_, b_: at(E, b_) / at(E, a_) - 1
    mixes = mix_rows(off, fE, 0.30, 'M_E50_30', E, end_cap, '主策略 70%＋0050 30%（窗初再平衡，未計再平衡成本）') + \
        mix_rows(off, fE, 0.70, 'C_E50_70', E, end_cap, '0050 70%＋主策略 30%（窗初再平衡，未計再平衡成本）') + \
        mix_rows(off, f2330, 0.30, 'M_2330_30', E, end_cap, '主策略 70%＋2330 30%（窗初再平衡，未計再平衡成本）')
    ext = external_rows(off, E, pxr)
    cols = ['line', 'anchor', 'start', 'end', 'status', 'n_j', 'ret', 'ret_0050', 'excess_pp', 'ret_stress', 'excess_stress_pp', 'cost', 'n_hold', 'source', 'note']
    FR = pd.DataFrame(rows + jm + E50 + mixes + ext).reindex(columns=cols)
    FR = FR.sort_values(['line', 'anchor'], na_position='last').reset_index(drop=True)
    FR.to_csv(out_dir / 'forward_returns.csv', index=False, float_format='%.10g')
    pd.DataFrame([dict(start=r['start'], end=r['end'], ret=r['ret']) for r in off], columns=['start', 'end', 'ret']).to_csv(out_dir / 'main_leg_returns.csv', index=False, float_format='%.10g')
    if diffs:
        f = out_dir / 'list_diffs.csv'
        old = pd.read_csv(f, dtype=str) if f.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(diffs).astype(str)]).drop_duplicates().to_csv(f, index=False)
    spec = json.loads(Path(a.spec).read_text())
    ntr = enrich_trades(out_dir, px, spec)
    dl = decision_log(out_dir, FR, Path(a.rule))
    head = ''
    try:
        head = subprocess.run(['git', '-C', str(Path(a.data).parent), 'log', '-1', '--format=%H %cI'], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    outs = {p.name: sha(p) for p in sorted(out_dir.glob('*.csv'))}
    rec = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), asof=str(asof.date()), start=a.start, data_head=head,
               script_sha256=sha(__file__), months=[str(m) for m in months], n_trades=ntr, decision=dl, outputs=outs, n_list_diffs=len(diffs))
    with open(out_dir / 'runs.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'outputs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
