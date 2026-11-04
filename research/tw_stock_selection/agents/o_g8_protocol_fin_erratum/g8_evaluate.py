#!/usr/bin/env python3
"""G8 前瞻協定的機械判定器（G8_PROTOCOL.json 的一部分；凍結後只准修 bug，規則不得改）。
只讀前瞻資料夾（預設 agents/live_tracking/main_forward），不寫入它；輸出寫到 --out（不得位於前瞻資料夾或 o_forward_infra_live 內）。

用法：
  python3 -I -B g8_evaluate.py status   [--forward DIR] [--first-anchor 2026-11] [--calib FILE ...] [--out FILE]
  python3 -I -B g8_evaluate.py calibrate [--forward DIR] [--prev FILE ...] --asof YYYY-MM-DD --out FILE
"""
import sys, json, math, argparse, hashlib, datetime
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
LIVE = AG / 'live_tracking/main_forward'
FORBID = [AG / 'live_tracking', AG / 'o_forward_infra_live']
PROTO = HERE / 'G8_PROTOCOL.json'
JL = [f'J{j}' for j in range(13)]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_proto(path):
    P = json.loads(Path(path).read_text(encoding='utf-8'))
    return P


def meta_of(fwd, line, anchor):
    nm = ('official' if line == 'MAIN_OFFICIAL' else line.lower()) + '_' + anchor.replace('-', '') + '.meta.json'
    p = Path(fwd) / 'paper_lists' / nm
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def void_reason(meta, P):
    """作廢規則（只看時戳與覆蓋率，不看報酬）。回傳 None＝有效。"""
    V = P['void_rules']
    if meta is None: return 'V0 快照不存在（上游過期且 fallback 未完成、或未凍結）'
    if not meta.get('ts_ok', False): return 'V1 快照凍結時間晚於成交日 09:00（ts_ok＝false）'
    try:
        if pd.Timestamp(meta['frozen_ts']) >= pd.Timestamp(meta['exec_open_ts']): return 'V1 frozen_ts ≥ exec_open_ts'
    except Exception:
        return 'V1 時戳無法解析'
    g = meta.get('rev_gate')
    if not isinstance(g, dict) or g.get('coverage') is None: return 'V2 快照沒有營收覆蓋率紀錄'
    if float(g['coverage']) < V['rev_coverage_min']: return f"V2 營收覆蓋率 {float(g['coverage']):.3f} < {V['rev_coverage_min']}"
    return None


def nav_turnover(fr, tr, line_fr, line_tr):
    """每窗換手（買＋賣成交值／窗初淨值；GROW 100 萬起算）。"""
    d = fr[(fr.line == line_fr)].sort_values('start')
    nav = 1e6; out = {}
    for _, r in d.iterrows():
        t = tr[(tr.line == line_tr) & (tr.date > r.start) & (tr.date <= r.end) & tr.kind.isin(['paper_whole', 'paper_odd'])]
        out[r.anchor] = float((t.shares * t.fill_px).sum() / nav) if len(t) else 0.0
        if r.status == 'complete' and np.isfinite(r.ret): nav *= 1 + r.ret
    return out


def cost_adjust(fr, tr, calibs):
    """主欄成本＝max(事前 K5T, 校準 p75)：每窗超額 −100·Δ·換手，只套用在訊號日晚於校準時戳的窗。"""
    adj = {}
    if not calibs: return adj
    for line_fr, line_tr in [(j, j) for j in JL] + [('MAIN_OFFICIAL', 'OFFICIAL')]:
        to = nav_turnover(fr, tr, line_fr, line_tr)
        for _, r in fr[fr.line == line_fr].iterrows():
            delta = 0.0
            for c in calibs:
                if pd.Timestamp(r.start) > pd.Timestamp(c['valid_from']): delta = c['delta_per_side']
            adj[(line_fr, r.anchor)] = 100.0 * delta * to.get(r.anchor, 0.0)
    return adj


def status(a):
    P = load_proto(a.protocol)
    R = P['rule']
    fwd = Path(a.forward)
    fr = pd.read_csv(fwd / 'forward_returns.csv')
    tr = pd.read_csv(fwd / 'trades.csv', dtype={'code': str}) if (fwd / 'trades.csv').exists() else pd.DataFrame(columns=['kind', 'line', 'date'])
    calibs = []
    for f in a.calib or []:
        c = json.loads(Path(f).read_text(encoding='utf-8'))
        body = {k: v for k, v in c.items() if k != 'sha256'}
        assert hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest() == c['sha256'], f'校準檔 {f} sha 不符'
        calibs.append(c)
    adj = cost_adjust(fr, tr, calibs)
    anchors = sorted(a_ for a_ in fr.anchor.unique() if a_ >= a.first_anchor)
    rows = []
    for an in anchors:
        J = fr[(fr.anchor == an) & fr.line.isin(JL)]
        closed = (len(J) == 13) and (J.status == 'complete').all()
        if not closed:
            rows.append(dict(anchor=an, state='open')); continue
        valid, voids = [], {}
        for _, r in J.iterrows():
            vr = void_reason(meta_of(fwd, r.line, an), P) if not a.ignore_void else None
            if vr: voids[r.line] = vr
            else: valid.append(r)
        if len(valid) < R['jmed_min_valid_j']:
            rows.append(dict(anchor=an, state='void', n_valid_j=len(valid), voids=voids)); continue
        ex = [float(r.excess_pp) - adj.get((r.line, an), 0.0) for r in valid]
        exs = [float(r.excess_stress_pp) for r in valid]
        rows.append(dict(anchor=an, state='valid', n_valid_j=len(valid), voids=voids, x=float(np.median(ex)), x_stress=float(np.median(exs)),
                         x_k5t_only=float(np.median([float(r.excess_pp) for r in valid])), j_p10=float(np.percentile(ex, 10))))
    # 序貫（與 OC 模擬同一套語意：先觸發者吸收；升級只在第 24 個有效月生效）
    sd, c, cd, b = P['params']['sigma_design'], P['params']['c'], P['params']['c_d'], P['params']['b']
    xs = []; st = 'FORWARD_VALIDATING'; events = []; sig_eff = c * sd; refit_on = False
    n_void_before12 = 0; n_tsflaw_first12 = 0; g8 = None
    for r in rows:
        if r['state'] == 'void':
            if len(xs) < 12: n_void_before12 += 1
            if r['anchor'] <= R['g8_window_last_anchor'] and any(v.startswith('V1') for v in r['voids'].values()): n_tsflaw_first12 += 1
            continue
        if r['state'] != 'valid': continue
        xs.append(r['x']); n = len(xs); S = float(np.sum(xs))
        look = n in (12, 18, 24) or (n > 24 and (n - 24) % 6 == 0)
        if look and n >= 12:
            s_hat = float(np.std(xs, ddof=1))
            if not refit_on and s_hat > R['refit_ratio'] * sd:
                refit_on = True; sig_eff = max(c * sd, s_hat); events.append(dict(n=n, anchor=r['anchor'], event='REFIT', sigma_eff=sig_eff))
        z = S / (sig_eff * math.sqrt(n))
        r.update(n=n, S=S, z=z)
        if st in ('FORWARD_FAILED', 'S1_EXCEPTION_EVIDENCE'): continue
        if n == 12:
            g8 = dict(n_void_before_12th_valid=n_void_before12, S12=S, passed=bool(S > 0),
                      o9_tsflaw_months=n_tsflaw_first12, o9_label=('時戳瑕疵，不判定（o9 口徑：≥ 2 個月）' if n_tsflaw_first12 >= 2 else None))
            if n_void_before12 > R['g8_max_void_in_12']:
                g8['label'] = '無法判定（作廢 > 2；延長到第 24 個有效月判一次）'
            elif S <= 0:
                st = 'FORWARD_FAILED'; events.append(dict(n=n, anchor=r['anchor'], event='LITERAL_G8_FAIL', S=S)); continue
        if n == 24 and g8 is not None and 'label' in g8:
            g8['extended'] = dict(S24=S, passed=bool(S > 0), n_void_total=sum(1 for q in rows if q['state'] == 'void'))
            if S <= 0 or g8['extended']['n_void_total'] > 2 * R['g8_max_void_in_12']:
                st = 'FORWARD_FAILED'; events.append(dict(n=n, anchor=r['anchor'], event='LITERAL_G8_FAIL_OR_UNDECIDABLE_EXT', S=S)); continue
        if st == 'FORWARD_VALIDATING' and look and z <= -cd:
            st = 'FORWARD_FAILED'; events.append(dict(n=n, anchor=r['anchor'], event='DOWN_BOUNDARY', z=z)); continue
        if st == 'FORWARD_VALIDATING' and n in R['up_looks'] and z >= b * math.sqrt(24 / n):
            st = 'UPGRADE_PENDING_24'; events.append(dict(n=n, anchor=r['anchor'], event='UPGRADE_BOUNDARY', z=z))
        if st == 'UPGRADE_PENDING_24' and n == 24:
            st = 'S1_EXCEPTION_EVIDENCE'; events.append(dict(n=n, anchor=r['anchor'], event='S1_EVIDENCE_AT_24'))
        elif st == 'FORWARD_VALIDATING' and n == 24:
            st = 'FORWARD_VALIDATING_NO_REJECTION_24'; events.append(dict(n=n, anchor=r['anchor'], event='NO_REJECTION_AT_24'))
        elif st == 'FORWARD_VALIDATING_NO_REJECTION_24' and look and z <= -cd:
            st = 'FORWARD_FAILED'; events.append(dict(n=n, anchor=r['anchor'], event='DOWN_BOUNDARY', z=z))
    lang = P['judgement_language']
    out = dict(protocol_sha256=sha(a.protocol), forward=str(fwd), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               state=st, state_text=lang['states'].get(st), g8_literal=g8, n_valid=len(xs), S=float(np.sum(xs)) if xs else 0.0,
               sigma_eff=sig_eff, refit=refit_on, events=events, anchors=rows, calibs=[c_['sha256'] for c_ in calibs],
               historical_gate_note=lang['historical_gate'])
    s = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    if a.out:
        op = Path(a.out).resolve()
        assert not any(str(op).startswith(str(f.resolve())) for f in FORBID), '輸出不得寫入前瞻資料夾'
        op.write_text(s, encoding='utf-8')
    print(s if not a.quiet else json.dumps(dict(state=st, n_valid=len(xs), events=events), ensure_ascii=False, default=str))
    return out


def calibrate(a):
    """成本校準：只用 2026-11-12 起的前瞻成交。資料門檻＝合格列 ≥ 40 且距 2026-11-12 ≥ 3 個月（取較晚）；每 12 個月最多一次。"""
    P = load_proto(a.protocol); CP = P['cost_calibration']
    fwd = Path(a.forward)
    tr = pd.read_csv(fwd / 'trades.csv', dtype={'code': str})
    asof = pd.Timestamp(a.asof); start = pd.Timestamp(CP['data_start'])
    for f in a.prev or []:
        c = json.loads(Path(f).read_text(encoding='utf-8'))
        assert (asof - pd.Timestamp(c['valid_from'])).days >= 365, '距上次校準不足 12 個月：不得重校'
    tr = tr[(pd.to_datetime(tr.date) >= start) & (pd.to_datetime(tr.date) <= asof)]
    sgn = np.where(tr.side.astype(str).str.lower().str.startswith('b'), 1.0, -1.0)
    real = tr[tr.kind == 'real'].copy(); real['slip'] = sgn[tr.kind.to_numpy() == 'real'] * (real.fill_px - real.open_px) / real.open_px
    paper = tr[tr.kind.isin(['paper_whole', 'paper_odd'])].copy()
    pw = paper[(paper.kind == 'paper_whole') & paper.share_of_open_vol.notna() & paper.share_of_open_vol_actual.notna() & paper.sigma60_prev.notna()]
    pw_d = CP['kappa'] * pw.sigma60_prev * (np.sqrt(pw.share_of_open_vol_actual) - np.sqrt(pw.share_of_open_vol))
    months_ok = asof >= start + pd.DateOffset(months=CP['min_months'])
    res = dict(asof=str(asof.date()), data_start=str(start.date()), n_real=int(len(real)), n_paper_whole=int(len(pw)), months_ok=bool(months_ok))
    d_real = d_open = 0.0
    if len(real) >= CP['min_rows'] and months_ok:
        # 模型對照：同日同檔同方向的紙上成交（官方日線）的模型滑價中位數
        m = paper.merge(real[['date', 'code', 'side']], on=['date', 'code', 'side'])
        ms = (np.where(m.side.str.lower().str.startswith('b'), 1, -1) * (m.fill_px - m.open_px) / m.open_px) if len(m) else pd.Series([0.0])
        d_real = max(0.0, float(np.percentile(real.slip, 75)) - float(np.median(ms)))
        res.update(real_slip_p50=float(np.median(real.slip)), real_slip_p75=float(np.percentile(real.slip, 75)), model_slip_p50_matched=float(np.median(ms)),
                   ticks_p50=float(real.ticks_from_open.median()), ticks_p75=float(real.ticks_from_open.quantile(0.75)),
                   odd_spread_p75=float(np.percentile(real.slip[real.is_odd == 1], 75)) if (real.is_odd == 1).any() else None)
    if len(pw) >= CP['min_rows'] and months_ok:
        d_open = max(0.0, float(np.percentile(pw_d, 75)))
        res.update(open_share_ratio_p50=float(np.median(pw.share_of_open_vol_actual / pw.share_of_open_vol)), open_impact_shortfall_p75=d_open)
    res['delta_real'] = d_real; res['delta_open'] = d_open
    res['performed'] = bool((len(real) >= CP['min_rows'] or len(pw) >= CP['min_rows']) and months_ok)
    res['delta_per_side'] = max(d_real, d_open) if res['performed'] else 0.0
    res['valid_from'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    res['input_sha256'] = sha(fwd / 'trades.csv'); res['protocol_sha256'] = sha(a.protocol)
    res['sha256'] = hashlib.sha256(json.dumps(res, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    op = Path(a.out).resolve()
    assert not any(str(op).startswith(str(f.resolve())) for f in FORBID), '輸出不得寫入前瞻資料夾'
    op.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['status', 'calibrate'])
    ap.add_argument('--forward', default=str(LIVE))
    ap.add_argument('--protocol', default=str(PROTO))
    ap.add_argument('--first-anchor', default='2026-11')
    ap.add_argument('--calib', nargs='*')
    ap.add_argument('--prev', nargs='*')
    ap.add_argument('--asof')
    ap.add_argument('--out')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--ignore-void', action='store_true', help='只供測試：忽略作廢規則（正式判定不得使用）')
    a = ap.parse_args()
    status(a) if a.cmd == 'status' else calibrate(a)
