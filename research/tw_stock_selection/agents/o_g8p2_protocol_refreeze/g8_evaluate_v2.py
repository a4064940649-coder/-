#!/usr/bin/env python3
"""G8P-2 前瞻協定的機械判定器（G8_PROTOCOL_v2.json 的一部分；凍結後只准修 bug，規則不得改；修 bug 須另立版本並重新外部錨定）。

唯一正式呼叫（判定）：   python3 -I -B g8_evaluate_v2.py status --official
唯一正式呼叫（校準登錄）：python3 -I -B g8_evaluate_v2.py calibrate --official
其他任何引數組合一律 NON_OFFICIAL（輸出最上方標記，不得引用）。測試用旗標只在 NON_OFFICIAL 下生效：
  --test-forward DIR --test-registry FILE --test-protocol FILE --test-manifest FILE --test-now ISO --test-skip-anchor --out FILE
協定、登錄簿、帳本路徑與名目起算月全部寫死；協定或本程式 sha 與 FREEZE_MANIFEST_v2.json 不符 → 拒絕（INVALID）。
只讀前瞻資料夾（live_tracking/main_forward），不寫入它與 o_forward_infra_live。依賴：numpy、pandas（不 import 研究程式）。
"""
import sys, json, math, hashlib, datetime, subprocess, argparse
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
PROTO = HERE / 'G8_PROTOCOL_v2.json'
MANIFEST = HERE / 'FREEZE_MANIFEST_v2.json'
REGISTRY = HERE / 'calib_registry.jsonl'
LEDGER = AG / 'live_tracking/main_forward'
BACKLOG = AG / 'BACKLOG.md'
RUNS = HERE / 'status_runs'
FORBID = [AG / 'live_tracking', AG / 'o_forward_infra_live']
NOMINAL_FIRST_ANCHOR = '2026-11'
DEADLINE_UTC = '2026-11-10T05:30:00+00:00'
TZ_TPE = datetime.timezone(datetime.timedelta(hours=8))
JL = [f'J{j}' for j in range(13)]
ZERO = '0' * 64
ORDER = ['FORWARD_FAILED', 'FORWARD_VALIDATING_NO_REJECTION_24', 'FORWARD_VALIDATING', 'UPGRADE_PENDING_24', 'S1_EXCEPTION_EVIDENCE']  # 由不利到有利


def sha_b(b): return hashlib.sha256(b).hexdigest()
def sha(p): return sha_b(Path(p).read_bytes())
def canon(o): return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
def utcnow(): return datetime.datetime.now(datetime.timezone.utc)


# ---------------------------------------------------------------- 完整性
def entry_sha(e):
    return sha_b(canon({k: v for k, v in e.items() if k != 'entry_sha'}).encode())


def read_registry(path, manifest):
    """讀全簿並驗雜湊鏈。回傳 (entries, problems)。"""
    probs = []
    if not Path(path).exists(): return [], ['REGISTRY_MISSING']
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    ents = []
    prev = ZERO
    for i, l in enumerate(lines):
        if not l.strip(): probs.append(f'REGISTRY_BLANK_LINE_{i}'); continue
        try: e = json.loads(l)
        except Exception: probs.append(f'REGISTRY_BAD_JSON_{i}'); continue
        if e.get('seq') != len(ents): probs.append(f'REGISTRY_SEQ_{i}')
        if e.get('prev_sha') != prev: probs.append(f'REGISTRY_CHAIN_BROKEN_{i}')
        if entry_sha(e) != e.get('entry_sha'): probs.append(f'REGISTRY_ENTRY_SHA_{i}')
        prev = e.get('entry_sha'); ents.append(e)
    if not ents or ents[0].get('type') != 'GENESIS': probs.append('REGISTRY_NO_GENESIS')
    elif manifest and ents[0].get('entry_sha') != manifest.get('registry_genesis_entry_sha'): probs.append('REGISTRY_GENESIS_MISMATCH')
    return ents, probs


def git_anchor(path):
    """外部錨定：檔案在 git 的最早 commit 時間，以及該 commit 的內容是否與現檔一致。"""
    try:
        top = subprocess.run(['git', '-C', str(HERE), 'rev-parse', '--show-toplevel'], capture_output=True, text=True, timeout=30).stdout.strip()
        rel = str(Path(path).resolve().relative_to(Path(top).resolve()))
        log = subprocess.run(['git', '-C', top, 'log', '--format=%H %cI', '--', rel], capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        if not log: return dict(ok=False, reason='NOT_COMMITTED')
        h, t = log[-1].split(' ', 1)
        blob = subprocess.run(['git', '-C', top, 'show', f'{h}:{rel}'], capture_output=True, timeout=60).stdout
        return dict(ok=sha_b(blob) == sha(path), first_commit=h, first_commit_utc=pd.Timestamp(t).tz_convert('UTC').isoformat(),
                    content_matches=sha_b(blob) == sha(path))
    except Exception as ex:
        return dict(ok=False, reason=f'GIT_ERROR {ex}')


def effective_first_anchor(anchor_utc):
    """凍結外部錨定（git 首次 commit）早於期限 → 2026-11；否則＝錨定後第一個完整錨定月（錨定時間早於該月 10 日 13:30 台北者）。"""
    t = pd.Timestamp(anchor_utc).tz_convert(TZ_TPE)
    if pd.Timestamp(anchor_utc) <= pd.Timestamp(DEADLINE_UTC): return NOMINAL_FIRST_ANCHOR
    m = pd.Period(t.strftime('%Y-%m'), 'M')
    cut = pd.Timestamp(m.year, m.month, 10, 13, 30, tz=TZ_TPE)
    return str(m if t < cut else m + 1)


# ---------------------------------------------------------------- 帳本
def meta_of(fwd, line, anchor):
    nm = ('official' if line == 'MAIN_OFFICIAL' else line.lower()) + '_' + anchor.replace('-', '') + '.meta.json'
    p = Path(fwd) / 'paper_lists' / nm
    return (json.loads(p.read_text(encoding='utf-8')), sha(p)) if p.exists() else (None, None)


def void_reason(meta, vr):
    if meta is None: return 'V0 快照不存在'
    if not meta.get('ts_ok', False): return 'V1 ts_ok＝false'
    try:
        if pd.Timestamp(meta['frozen_ts']) >= pd.Timestamp(meta['exec_open_ts']): return 'V1 frozen_ts ≥ exec_open_ts'
    except Exception:
        return 'V1 時戳無法解析'
    g = meta.get('rev_gate')
    if not isinstance(g, dict) or g.get('coverage') is None: return 'V2 沒有營收覆蓋率（含 J0_UNVERIFIED）'
    if float(g['coverage']) < vr['rev_coverage_min']: return f"V2 營收覆蓋率 {float(g['coverage']):.3f} < {vr['rev_coverage_min']}"
    return None


def turnover(fr, tr, line):
    """每窗換手（買＋賣成交值／窗初淨值；GROW 100 萬起算）。"""
    d = fr[fr.line == line].sort_values('start')
    nav = 1e6; out = {}
    for _, r in d.iterrows():
        t = tr[(tr.line == line) & (tr.date > r.start) & (tr.date <= r.end) & tr.kind.isin(['paper_whole', 'paper_odd'])] if len(tr) else tr
        out[r.anchor] = float((t.shares.astype(float) * t.fill_px.astype(float)).sum() / nav) if len(t) else 0.0
        if r.status == 'complete' and np.isfinite(float(r.ret)): nav *= 1 + float(r.ret)
    return out


def month_rows(fwd, P, first_anchor, calibs, mode):
    """逐錨定月：有效 J 窗的 x_w＝min(excess_pp − adj, excess_stress_pp)，有效窗 ≥ jmed_min_valid_j 取中位數。
    mode＝'retro'：REAL_CALIB 套用到所有窗；'forward'：只套用到窗起點晚於 valid_from 的窗。"""
    fr = pd.read_csv(Path(fwd) / 'forward_returns.csv')
    trp = Path(fwd) / 'trades.csv'
    tr = pd.read_csv(trp, dtype={'code': str}) if trp.exists() else pd.DataFrame(columns=['kind', 'line', 'date', 'shares', 'fill_px'])
    R = P['rule']; VR = P['void_rules']
    to = {j: turnover(fr, tr, j) for j in JL} if calibs else {}
    rows, meta_shas = [], {}
    for an in sorted(a for a in fr.anchor.astype(str).unique() if a >= first_anchor):
        J = fr[(fr.anchor.astype(str) == an) & fr.line.isin(JL)]
        if not ((len(J) == 13) and (J.status == 'complete').all()):
            rows.append(dict(anchor=an, state='open')); continue
        valid, voids = [], {}
        for _, r in J.iterrows():
            m, h = meta_of(fwd, r.line, an); meta_shas[f'{r.line}_{an}'] = h
            v = void_reason(m, VR)
            if v: voids[r.line] = v
            else: valid.append(r)
        if len(valid) < R['jmed_min_valid_j']:
            rows.append(dict(anchor=an, state='void', n_valid_j=len(valid), voids=voids)); continue
        xs = []
        for r in valid:
            adj = 0.0
            for c in calibs:
                if mode == 'retro' or pd.Timestamp(r.start) > pd.Timestamp(c['valid_from']).tz_convert('UTC').tz_localize(None):
                    adj = 100.0 * float(c['delta_per_side']) * to.get(r.line, {}).get(an, 0.0)
            xs.append(min(float(r.excess_pp) - adj, float(r.excess_stress_pp)))
        rows.append(dict(anchor=an, state='valid', n_valid_j=len(valid), voids=voids, x=float(np.median(xs)),
                         x_normal_med=float(np.median([float(r.excess_pp) for r in valid])), x_stress_med=float(np.median([float(r.excess_stress_pp) for r in valid])),
                         j_p10=float(np.percentile(xs, 10))))
    return rows, meta_shas


def sequential(rows, P):
    """序貫判定（V1 語意；OC 模擬同一套）。"""
    pr = P['params']; R = P['rule']
    sd, c, cd, b, dl = pr['sigma_design'], pr['c'], pr['c_d'], pr['b'], pr.get('delta', 0.0)
    upg = pr.get('upgrade', True)
    xs = []; st = 'FORWARD_VALIDATING'; events = []; g8 = None; nvb12 = 0
    for r in rows:
        if r['state'] == 'void':
            if len(xs) < 12: nvb12 += 1
            continue
        if r['state'] != 'valid': continue
        xs.append(r['x']); n = len(xs); S = float(np.sum(xs)); Sd = S - dl * n
        look = n in (12, 18, 24) or (n > 24 and (n - 24) % 6 == 0)
        if look:
            s_n = float(np.std(np.array(xs) - dl, ddof=1))
            sig = max(c * sd, s_n); z = Sd / (sig * math.sqrt(n))
            r.update(n=n, S=S, s_n=s_n, sigma_eff=sig, z=z)
        else:
            r.update(n=n, S=S)
        if st in ('FORWARD_FAILED', 'S1_EXCEPTION_EVIDENCE'): continue
        if n == 12:
            g8 = dict(n_void_before_12th_valid=nvb12, S12=S, passed=bool(S > 0))
            if nvb12 > R['g8_max_void_in_12']:
                g8['label'] = '無法判定（作廢 > 2；延長到第 24 個有效月判一次）'
            elif S <= 0:
                st = 'FORWARD_FAILED'; events.append(dict(n=n, anchor=r['anchor'], event='LITERAL_G8_FAIL', S=S)); continue
        if n == 24 and g8 is not None and 'label' in g8:
            nv = sum(1 for q in rows if q['state'] == 'void')
            g8['extended'] = dict(S24=S, passed=bool(S > 0), n_void_total=nv)
            if S <= 0 or nv > 2 * R['g8_max_void_in_12']:
                st = 'FORWARD_FAILED'; events.append(dict(n=n, anchor=r['anchor'], event='LITERAL_G8_FAIL_OR_UNDECIDABLE_EXT', S=S)); continue
        if not look: continue
        if st == 'FORWARD_VALIDATING' and z <= -cd:
            st = 'FORWARD_FAILED'; events.append(dict(n=n, anchor=r['anchor'], event='DOWN_BOUNDARY', z=z)); continue
        if upg and st == 'FORWARD_VALIDATING' and n in (12, 24) and z >= b * math.sqrt(24 / n):
            st = 'UPGRADE_PENDING_24'; events.append(dict(n=n, anchor=r['anchor'], event='UPGRADE_BOUNDARY', z=z, sigma_check=sigma_check(xs, P)))
        if st == 'UPGRADE_PENDING_24' and n == 24:
            st = 'S1_EXCEPTION_EVIDENCE'; events.append(dict(n=n, anchor=r['anchor'], event='S1_EVIDENCE_AT_24'))
        elif st == 'FORWARD_VALIDATING' and n == 24:
            st = 'FORWARD_VALIDATING_NO_REJECTION_24'; events.append(dict(n=n, anchor=r['anchor'], event='NO_REJECTION_AT_24'))
        elif st == 'FORWARD_VALIDATING_NO_REJECTION_24' and n > 24 and z <= -cd:
            st = 'FORWARD_FAILED'; events.append(dict(n=n, anchor=r['anchor'], event='DOWN_BOUNDARY', z=z))
    return st, events, g8, xs


def sigma_check(xs, P):
    x = np.asarray(xs, float)
    s_n = float(np.std(x, ddof=1))
    r12 = float(np.std(x[-12:], ddof=1)) if len(x) >= 12 else None
    r24 = float(np.std(x[-24:], ddof=1)) if len(x) >= 24 else None
    grid = P['oc']['sigma_grid']; lab = P['oc']['sigma_label']
    chk = max(v for v in (s_n, r12, r24) if v is not None)
    cell = next((l for g, l in zip(grid, lab) if g >= chk - 1e-12), None)
    fu = P['oc']['false_up_max_by_sigma'].get(cell) if cell else None
    return dict(sample_sd=s_n, roll12_sd=r12, roll24_sd=r24, sigma_checked=chk, sigma_cell=cell or 'OUT_OF_GRID',
                cell_false_up_max_over_dgp_s=fu, warning=None if cell else 'σ 超出 OC 格（> 10.4）：升級證據附缺陷，OC 未涵蓋')


def schedule(until):
    d = pd.Timestamp('2027-02-12'); out = []
    while d <= until:
        out.append(str(d.date())); d = d + pd.DateOffset(months=3)
    return out


# ---------------------------------------------------------------- status
def status(a, official, argv):
    now = pd.Timestamp(a.test_now) if (a.test_now and not official) else pd.Timestamp(utcnow())
    if now.tzinfo is None: now = now.tz_localize('UTC')
    proto_p = Path(a.test_protocol) if (a.test_protocol and not official) else PROTO
    man_p = Path(a.test_manifest) if (a.test_manifest and not official) else MANIFEST
    reg_p = Path(a.test_registry) if (a.test_registry and not official) else REGISTRY
    fwd = Path(a.test_forward) if (a.test_forward and not official) else LEDGER
    problems = []
    man = json.loads(man_p.read_text(encoding='utf-8')) if man_p.exists() else None
    if man is None: problems.append('MANIFEST_MISSING')
    if not proto_p.exists():
        out = dict(OFFICIAL=False, official_call=bool(official), label='INVALID（不得引用）', argv=argv, flags={k: v for k, v in vars(a).items()},
                   utc=utcnow().isoformat(), problems=problems + ['PROTOCOL_MISSING'], state=None, state_if_valid=None)
        if not a.quiet: print(json.dumps(out, ensure_ascii=False, indent=1))
        return out
    P = json.loads(proto_p.read_text(encoding='utf-8'))
    if man and sha(proto_p) != man.get('protocol_sha256'): problems.append('PROTOCOL_SHA_MISMATCH')
    if man and sha(__file__) != man.get('evaluator_sha256'): problems.append('EVALUATOR_SHA_MISMATCH')
    ents, rp = read_registry(reg_p, man); problems += rp
    due = [d for d in schedule(now.tz_localize(None) - pd.Timedelta(days=30))]
    have = {e.get('sched_date') for e in ents}
    missing = [d for d in due if d not in have]
    if missing: problems.append(f'CALIB_MISSING {missing}')
    # 外部錨定
    if official or not a.test_skip_anchor:
        bl_ok = bool(man) and BACKLOG.exists() and (man_p.exists() and sha(man_p) in BACKLOG.read_text(encoding='utf-8'))
        ga = git_anchor(man_p) if man_p.exists() else dict(ok=False, reason='NO_MANIFEST')
        anchor = dict(backlog_ok=bl_ok, git=ga)
        if not bl_ok: problems.append('ANCHOR_BACKLOG_MISSING')
        if not ga.get('ok'): problems.append('ANCHOR_GIT_MISSING')
        eff = effective_first_anchor(ga['first_commit_utc']) if ga.get('ok') else None
    else:
        anchor = dict(skipped=True); eff = a.test_first_anchor or NOMINAL_FIRST_ANCHOR
    calibs = [e for e in ents if e.get('type') == 'REAL_CALIB' and e.get('applied')]
    res = {}
    if eff is not None:
        for mode in ('retro', 'forward'):
            rows, mshas = month_rows(fwd, P, eff, calibs, mode)
            stt, ev, g8, xs = sequential(rows, P)
            res[mode] = dict(state=stt, events=ev, g8_literal=g8, n_valid=len(xs), S=float(np.sum(xs)) if xs else 0.0, anchors=rows)
        state = min((res['retro']['state'], res['forward']['state']), key=ORDER.index)
    else:
        state = None; mshas = {}
    invalid = bool(problems)
    lang = P['judgement_language']
    led = {f: (sha(fwd / f) if (fwd / f).exists() else None) for f in ('forward_returns.csv', 'trades.csv', 'fills_input.csv')}
    out = dict(OFFICIAL=bool(official and not invalid), official_call=bool(official),
               label=('OFFICIAL' if official and not invalid else ('INVALID（不得引用）' if official else 'NON_OFFICIAL（不得引用）')),
               argv=argv, flags={k: v for k, v in vars(a).items()}, utc=utcnow().isoformat(), now_used=now.isoformat(),
               protocol_path=str(proto_p), protocol_sha256=sha(proto_p), manifest_sha256=(sha(man_p) if man_p.exists() else None),
               evaluator_sha256=sha(__file__), registry_path=str(reg_p), registry_sha256=(sha(reg_p) if reg_p.exists() else None),
               registry_n_entries=len(ents), registry_types=[e.get('type') for e in ents], calibs_applied=[e.get('entry_sha') for e in calibs],
               ledger=str(fwd), ledger_sha256=led, meta_sha256_combined=sha_b(canon(mshas).encode()), n_meta=len([v for v in mshas.values() if v]),
               anchor=anchor, effective_first_anchor=eff, problems=problems,
               state=(None if invalid else state), state_if_valid=state, state_text=(lang['states'].get(state) if state and not invalid else lang['invalid']),
               by_mode=res, scope=P['scope']['s1_evidence_scope'], historical_gate_note=lang['historical_gate'])
    s = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    if official:
        RUNS.mkdir(exist_ok=True)
        op = RUNS / f"status_{utcnow().strftime('%Y%m%dT%H%M%SZ')}_{'OFFICIAL' if not invalid else 'INVALID'}.json"
    else:
        op = Path(a.out) if a.out else None
    if op is not None:
        op = op.resolve()
        assert not any(str(op).startswith(str(f.resolve())) for f in FORBID), '輸出不得寫入前瞻資料夾'
        op.write_text(s, encoding='utf-8')
    if not a.quiet: print(s)
    return out


# ---------------------------------------------------------------- calibrate
def calibrate(a, official, argv):
    now = pd.Timestamp(a.test_now) if (a.test_now and not official) else pd.Timestamp(utcnow())
    if now.tzinfo is None: now = now.tz_localize('UTC')
    proto_p = Path(a.test_protocol) if (a.test_protocol and not official) else PROTO
    man_p = Path(a.test_manifest) if (a.test_manifest and not official) else MANIFEST
    reg_p = Path(a.test_registry) if (a.test_registry and not official) else REGISTRY
    fwd = Path(a.test_forward) if (a.test_forward and not official) else LEDGER
    P = json.loads(proto_p.read_text(encoding='utf-8')); CP = P['cost_calibration']
    man = json.loads(man_p.read_text(encoding='utf-8')) if man_p.exists() else None
    ents, probs = read_registry(reg_p, man)
    if man and sha(proto_p) != man.get('protocol_sha256'): probs.append('PROTOCOL_SHA_MISMATCH')
    if man and sha(__file__) != man.get('evaluator_sha256'): probs.append('EVALUATOR_SHA_MISMATCH')
    if probs:
        print(json.dumps(dict(label='INVALID', problems=probs), ensure_ascii=False)); return dict(label='INVALID', problems=probs)
    have = {e.get('sched_date') for e in ents}
    due = [d for d in schedule(now.tz_localize(None)) if d not in have]
    if not due:
        print(json.dumps(dict(label='NO_CALIBRATION_DUE', now=now.isoformat()), ensure_ascii=False)); return dict(label='NO_CALIBRATION_DUE')
    sd = pd.Timestamp(due[0])
    trp = fwd / 'trades.csv'
    tr = pd.read_csv(trp, dtype={'code': str}) if trp.exists() else pd.DataFrame(columns=['kind', 'date'])
    tr = tr[(pd.to_datetime(tr.date) >= pd.Timestamp(CP['data_start'])) & (pd.to_datetime(tr.date) <= sd)] if len(tr) else tr
    real = tr[tr.kind == 'real'].copy() if len(tr) else tr
    n_real = int(len(real)); months_real = int(pd.to_datetime(real.date).dt.to_period('M').nunique()) if n_real else 0
    span_ok = bool(n_real and (sd - pd.to_datetime(real.date).min()).days >= 90)
    real_ok = n_real >= CP['min_rows'] and months_real >= CP['min_months'] and span_ok
    pw = tr[(tr.kind == 'paper_whole')] if len(tr) else tr
    if len(pw): pw = pw[pw.share_of_open_vol.notna() & pw.share_of_open_vol_actual.notna() & pw.sigma60_prev.notna()]
    d_open = None
    if len(pw) >= CP['min_rows']:
        d_open = max(0.0, float(np.percentile(CP['kappa'] * pw.sigma60_prev * (np.sqrt(pw.share_of_open_vol_actual) - np.sqrt(pw.share_of_open_vol)), 75)))
    d_real = None
    if real_ok:
        paper = tr[tr.kind.isin(['paper_whole', 'paper_odd'])]
        sg = lambda df: np.where(df.side.astype(str).str.lower().str.startswith('b'), 1.0, -1.0)
        real = real.assign(slip=sg(real) * (real.fill_px - real.open_px) / real.open_px)
        m = paper.merge(real[['date', 'code', 'side']], on=['date', 'code', 'side'])
        ms = (sg(m) * (m.fill_px - m.open_px) / m.open_px) if len(m) else np.array([0.0])
        d_real = max(0.0, float(np.percentile(real.slip, 75)) - float(np.median(ms)))
    last_real = [e for e in ents if e.get('type') == 'REAL_CALIB']
    slot_open = not last_real or (sd - pd.Timestamp(last_real[-1]['sched_date'])).days >= 365
    if real_ok and slot_open:
        typ = 'REAL_CALIB'; delta = max(d_real, d_open or 0.0)
    elif real_ok:
        typ = 'REAL_SLOT_LOCKED'; delta = 0.0
    elif d_open is not None:
        typ = 'DELTA_OPEN_ONLY'; delta = 0.0
    else:
        typ = 'NO_REAL_FILLS'; delta = 0.0
    e = dict(seq=len(ents), type=typ, sched_date=str(sd.date()), utc=utcnow().isoformat(), valid_from=utcnow().isoformat(),
             counts_toward_slot=(typ == 'REAL_CALIB'), applied=(typ == 'REAL_CALIB'), delta_per_side=float(delta),
             delta_real=d_real, delta_open=d_open, delta_open_desc=CP['delta_open_desc'], n_real=n_real, n_real_months=months_real,
             n_paper_whole=int(len(pw)), input_trades_sha256=(sha(trp) if trp.exists() else None), protocol_sha256=sha(proto_p),
             evaluator_sha256=sha(__file__), official_call=bool(official), argv=argv, prev_sha=(ents[-1]['entry_sha'] if ents else ZERO))
    e['entry_sha'] = entry_sha(e)
    if official or a.test_registry:
        with open(reg_p, 'a', encoding='utf-8') as fh: fh.write(canon(e) + '\n')
    if not a.quiet: print(json.dumps(dict(label=('OFFICIAL' if official else 'NON_OFFICIAL（不得引用）'), entry=e), ensure_ascii=False))
    return e


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['status', 'calibrate'])
    ap.add_argument('--official', action='store_true')
    for f in ('--test-forward', '--test-registry', '--test-protocol', '--test-manifest', '--test-now', '--test-first-anchor', '--out'):
        ap.add_argument(f)
    ap.add_argument('--test-skip-anchor', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args(argv)
    official = argv in (['status', '--official'], ['calibrate', '--official'])
    return (status if a.cmd == 'status' else calibrate)(a, official, argv)


if __name__ == '__main__':
    main()
