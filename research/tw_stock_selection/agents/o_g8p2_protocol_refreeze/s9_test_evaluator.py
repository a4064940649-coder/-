#!/usr/bin/env python3
"""步驟 9：判定器一致性與旗標測試（合成帳本，全部在 scratchpad；不碰前瞻資料夾）。
 A. ≥ 200 條合成路徑：判定器 vs OC 向量化程式（g8lib2.outcomes，24 個有效月）vs 本檔獨立純量參考實作（任意長度、含作廢延長）→ 0 不一致
 B. 旗標／完整性：非正式旗標 → NON_OFFICIAL；登錄簿竄改、漏登、協定竄改 → INVALID；P9（沒有真錢成交 24 個月只有不計名額條目）；REAL_CALIB 名額；生效錨定月規則
用法：python3 -I -B s9_test_evaluator.py --dir <協定所在資料夾（草稿或本資料夾）> [--record]"""
import sys, json, math, shutil, argparse, importlib.util, copy
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import g8lib2 as L

ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--record', action='store_true'); A = ap.parse_args()
PD = Path(A.dir)
spec = importlib.util.spec_from_file_location('g8ev2', HERE / 'g8_evaluate_v2.py'); EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
PROTO = json.loads((PD / 'G8_PROTOCOL_v2.json').read_text(encoding='utf-8'))
PR = PROTO['params']
TMP = L.SCR / 'evtest'
shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
REG0 = TMP / 'reg0.jsonl'; shutil.copy(PD / 'calib_registry.jsonl', REG0)
BASEARGS = ['--test-protocol', str(PD / 'G8_PROTOCOL_v2.json'), '--test-manifest', str(PD / 'FREEZE_MANIFEST_v2.json'), '--test-skip-anchor', '--quiet']
TRADE_COLS = ['kind', 'line', 'date', 'code', 'side', 'is_odd', 'shares', 'fill_px', 'open_px', 'prev_close', 'close_px', 'ticks_from_open', 'sigma60_prev', 'V60_prev',
              'dec', 'oa_share_dec', 'OA_pred', 'share_of_open_vol', 'share_of_open_vol_actual', 'in_disposal', 'event_id', 'cost_rate', 'share_of_qref', 'order_time', 'limit_px']


# ---------------------------------------------------------------- 合成帳本
def make_ledger(d, months, rng):
    """months：list of dict(anchor, x_target, void_month(bool))。回傳期望的逐月 (state, x)。"""
    shutil.rmtree(d, ignore_errors=True); (d / 'paper_lists').mkdir(parents=True)
    rows, expect = [], []
    for m in months:
        an = m['anchor']; per = pd.Period(an, 'M')
        xs = []
        nvalid = 0
        void_lines = set()
        if m['void_month']:
            void_lines = set(rng.choice(13, size=7, replace=False).tolist())
        for j in range(13):
            st = pd.Timestamp(per.year, per.month, 10) + pd.Timedelta(days=j)
            en = st + pd.Timedelta(days=30)
            ex = m['x_target'] + rng.normal(0, 2.0)
            exs = ex - rng.uniform(-0.2, 1.0)
            rows.append(dict(line=f'J{j}', anchor=an, start=str(st.date()), end=str(en.date()), status='complete', n_j=1, ret=0.01, ret_0050=0.0,
                             excess_pp=ex, ret_stress=0.0, excess_stress_pp=exs, ret_cap10=0.0, excess_cap10_pp=ex, cost=0.0, n_hold=10, source='synthetic', note=''))
            meta = dict(ts_ok=True, frozen_ts=f'{st.date()}T06:49:00+08:00', exec_open_ts=f'{(st + pd.Timedelta(days=1)).date()}T09:00:00+08:00',
                        rev_gate=dict(status='PASS', coverage=1.0))
            kind = 'ok'
            if j == 0: meta['rev_gate'] = dict(status='J0_UNVERIFIED'); kind = 'void'          # J0_UNVERIFIED → V2 作廢
            elif j in void_lines: kind = 'nometa'
            elif rng.random() < 0.05: meta['ts_ok'] = False; kind = 'void'
            elif rng.random() < 0.03: meta['rev_gate'] = dict(status='LOW', coverage=0.9); kind = 'void'
            if kind != 'nometa':
                (d / 'paper_lists' / f'j{j}_{per.strftime("%Y%m")}.meta.json').write_text(json.dumps(meta), encoding='utf-8')
            if kind == 'ok': xs.append(min(ex, exs))
        if len(xs) >= 7: expect.append(('valid', float(np.median(xs))))
        else: expect.append(('void', None))
    pd.DataFrame(rows).to_csv(d / 'forward_returns.csv', index=False)
    pd.DataFrame(columns=TRADE_COLS).to_csv(d / 'trades.csv', index=False)
    return expect


# ---------------------------------------------------------------- 獨立純量參考實作（依協定文字）
def ref_state(expect, P):
    sd, c, cd, b, dl = P['sigma_design'], P['c'], P['c_d'], P['b'], P['delta']
    xs = []; st = 'FORWARD_VALIDATING'; nvb = 0; ext = False
    for kind, x in expect:
        if kind == 'void':
            if len(xs) < 12: nvb += 1
            continue
        xs.append(x); n = len(xs)
        if st in ('FORWARD_FAILED', 'S1_EXCEPTION_EVIDENCE'): continue
        if n == 12:
            if nvb > 2: ext = True
            elif sum(xs) <= 0: st = 'FORWARD_FAILED'; continue
        if n == 24 and ext:
            nv = sum(1 for k, _ in expect if k == 'void')
            if sum(xs) <= 0 or nv > 4: st = 'FORWARD_FAILED'; continue
        look = n in (12, 18, 24) or (n > 24 and (n - 24) % 6 == 0)
        if not look: continue
        y = [v - dl for v in xs]
        m = sum(y) / n; s = math.sqrt(sum((v - m) ** 2 for v in y) / (n - 1))
        z = sum(y) / (max(c * sd, s) * math.sqrt(n))
        if st == 'FORWARD_VALIDATING' and z <= -cd: st = 'FORWARD_FAILED'; continue
        if st == 'FORWARD_VALIDATING' and n in (12, 24) and z >= b * math.sqrt(24 / n): st = 'UPGRADE_PENDING_24'
        if st == 'UPGRADE_PENDING_24' and n == 24: st = 'S1_EXCEPTION_EVIDENCE'
        elif st == 'FORWARD_VALIDATING' and n == 24: st = 'FORWARD_VALIDATING_NO_REJECTION_24'
        elif st == 'FORWARD_VALIDATING_NO_REJECTION_24' and n > 24 and z <= -cd: st = 'FORWARD_FAILED'
    return st


def lib_state(xs24, P):
    spec_ = dict(stat=P['stat'], delta=P['delta'])
    o = L.outcomes(L.zstats(np.asarray(xs24, float)[None, :], spec_), P['b'], P['c_d'], upgrade=P['upgrade'])
    return 'S1_EXCEPTION_EVIDENCE' if o['up'][0] else ('FORWARD_FAILED' if o['fail'][0] else 'FORWARD_VALIDATING_NO_REJECTION_24')


def run_status(fwd, reg, extra=()):
    return EV.main(['status', '--test-forward', str(fwd), '--test-registry', str(reg)] + BASEARGS + list(extra))


# ---------------------------------------------------------------- A：一致性
rng = np.random.default_rng(20261108)
results = []; mism = []
FWD = TMP / 'fwd'
NPATH = 260
for i in range(NPATH):
    mu = [-1.5, -0.5, 0.0, 1.0, 2.5, 4.0][i % 6]; sg = [6.14, 8.4, 10.4][(i // 6) % 3]
    long_ = i >= 220                      # 40 條長路徑（36 個有效月、可含 > 2 作廢）
    target_valid = 36 if long_ else 24
    maxvoid12 = 4 if long_ else 2
    months = []; per = pd.Period('2026-11', 'M'); nval = 0; nv12 = 0
    while nval < target_valid:
        vm = rng.random() < 0.08 and (nval >= 12 or nv12 < maxvoid12)
        if vm and nval < 12: nv12 += 1
        months.append(dict(anchor=str(per), x_target=float(mu + sg * rng.standard_normal()), void_month=bool(vm)))
        per += 1
        if not vm: nval += 1
    exp = make_ledger(FWD, months, rng)
    # 期望月份有效性可能因 J 作廢與 V0 而與 void_month 不同 → 以 exp 為準重數有效月；不足 target 就再補
    out = run_status(FWD, REG0)
    ev = out['by_mode']['retro']; evf = out['by_mode']['forward']
    ex_valid = [x for k, x in exp if k == 'valid']
    ev_valid = [r['x'] for r in ev['anchors'] if r['state'] == 'valid']
    rs = ref_state(exp, PR)
    nvb = 0; cnt = 0
    for k, _ in exp:
        if k == 'void' and cnt < 12: nvb += 1
        if k == 'valid': cnt += 1
    ls = lib_state(ex_valid[:24], PR) if (len(ex_valid) >= 24 and nvb <= 2) else None
    if ls is not None and len(ex_valid) > 24 and rs != ls and rs == 'FORWARD_FAILED' and ls == 'FORWARD_VALIDATING_NO_REJECTION_24':
        ls_cmp = 'post24'                  # 24 個月後的下修檢視（OC 只模擬到 24）：只比對參考實作
    else:
        ls_cmp = ls
    ok = (len(ex_valid) == len(ev_valid) and np.allclose(ex_valid, ev_valid, atol=1e-9) and out['state_if_valid'] == rs == ev['state'] == evf['state']
          and (ls_cmp in (None, 'post24') or ls_cmp == rs))
    results.append(dict(i=i, mu=mu, sigma=sg, n_valid=len(ex_valid), n_void_before12=nvb, evaluator=ev['state'], reference=rs, lib24=ls, ok=bool(ok), label=out['label']))
    if not ok: mism.append(results[-1])
cnt_states = pd.Series([r['evaluator'] for r in results]).value_counts().to_dict()
A_res = dict(n_paths=NPATH, n_mismatch=len(mism), mismatches=mism[:10], state_counts=cnt_states,
             n_with_lib_compare=int(sum(r['lib24'] is not None for r in results)), n_ext_void=int(sum(r['n_void_before12'] > 2 for r in results)),
             labels=sorted({r['label'] for r in results}))
print('A', json.dumps({k: v for k, v in A_res.items() if k != 'mismatches'}, ensure_ascii=False), mism[:3])

# ---------------------------------------------------------------- B：旗標與完整性
B = {}
# B1 非正式旗標
o = EV.main(['status', '--official', '--quiet', '--test-forward', str(FWD)] + BASEARGS)
B['B1_extra_flags_non_official'] = dict(label=o['label'], OFFICIAL=o['OFFICIAL'], ok=(o['OFFICIAL'] is False and o['label'].startswith('NON_OFFICIAL')))
# B2 登錄簿竄改
reg_t = TMP / 'reg_tamper.jsonl'
e0 = json.loads(REG0.read_text().splitlines()[0]); e0['note'] = 'tampered'
reg_t.write_text(EV.canon(e0) + '\n')
o = run_status(FWD, reg_t)
B['B2_registry_tamper'] = dict(problems=o['problems'], ok=any('REGISTRY' in p for p in o['problems']) and o['state'] is None)
# B3 鏈斷（追加一筆 prev_sha 錯誤）
reg_b = TMP / 'reg_break.jsonl'
e1 = dict(seq=1, type='NO_REAL_FILLS', sched_date='2027-02-12', prev_sha='f' * 64); e1['entry_sha'] = EV.entry_sha(e1)
reg_b.write_text(REG0.read_text() + EV.canon(e1) + '\n')
o = run_status(FWD, reg_b)
B['B3_chain_break'] = dict(problems=o['problems'], ok=any('CHAIN' in p for p in o['problems']))
# B4 漏登
o = run_status(FWD, REG0, ['--test-now', '2027-04-01T00:00:00Z'])
B['B4_calib_missing'] = dict(problems=o['problems'], ok=any('CALIB_MISSING' in p for p in o['problems']))
o = run_status(FWD, REG0, ['--test-now', '2027-03-10T00:00:00Z'])
B['B4b_within_grace'] = dict(problems=o['problems'], ok=not o['problems'])
# B5 協定竄改
pt = TMP / 'proto_t.json'; pj = copy.deepcopy(PROTO); pj['params']['b'] = 1.0; pt.write_text(json.dumps(pj, ensure_ascii=False))
o = EV.main(['status', '--test-forward', str(FWD), '--test-registry', str(REG0), '--test-protocol', str(pt), '--test-manifest', str(PD / 'FREEZE_MANIFEST_v2.json'), '--test-skip-anchor', '--quiet'])
B['B5_protocol_tamper'] = dict(problems=o['problems'], ok='PROTOCOL_SHA_MISMATCH' in o['problems'] and o['state'] is None)
# B6 P9：沒有真錢成交 24 個月（紙上整股有、零股有）→ 只有不計名額條目
FW2 = TMP / 'fwd_p9'; shutil.copytree(FWD, FW2)
tr = []
for k in range(80):
    d = (pd.Timestamp('2026-11-12') + pd.Timedelta(days=9 * k)).date()
    tr.append(dict(kind='paper_whole', line='J1', date=str(d), code='2330', side='buy', is_odd=0, shares=1000, fill_px=100.0, open_px=100.0, sigma60_prev=0.02,
                   share_of_open_vol=0.01, share_of_open_vol_actual=0.012))
    tr.append(dict(kind='paper_odd', line='J1', date=str(d), code='2330', side='buy', is_odd=1, shares=50, fill_px=100.2, open_px=100.0))
pd.DataFrame(tr).reindex(columns=TRADE_COLS).to_csv(FW2 / 'trades.csv', index=False)
reg9 = TMP / 'reg_p9.jsonl'; shutil.copy(REG0, reg9)
types = []
for sd_ in EV.schedule(pd.Timestamp('2028-11-12')):
    e = EV.main(['calibrate', '--test-forward', str(FW2), '--test-registry', str(reg9), '--test-protocol', str(PD / 'G8_PROTOCOL_v2.json'),
                 '--test-manifest', str(PD / 'FREEZE_MANIFEST_v2.json'), '--test-now', f'{sd_}T08:00:00Z', '--quiet'])
    types.append((sd_, e['type'], e['counts_toward_slot'], e['applied']))
o = run_status(FW2, reg9, ['--test-now', '2028-12-31T00:00:00Z'])
B['B6_P9_no_real_fills'] = dict(entries=types, status_problems=o['problems'], calibs_applied=o['calibs_applied'],
                                ok=all(t[1] in ('NO_REAL_FILLS', 'DELTA_OPEN_ONLY') and not t[2] and not t[3] for t in types) and not o['problems'] and not o['calibs_applied'])
# B7 REAL_CALIB：60 筆真錢成交跨 4 個月
FW3 = TMP / 'fwd_real'; shutil.copytree(FW2, FW3)
tr3 = pd.read_csv(FW3 / 'trades.csv', dtype={'code': str})
real = [dict(kind='real', line='OFFICIAL', date=str((pd.Timestamp('2026-11-12') + pd.Timedelta(days=2 * k)).date()), code='2330', side='buy', is_odd=0, shares=1000,
             fill_px=100.3, open_px=100.0) for k in range(60)]
pd.concat([tr3, pd.DataFrame(real)]).reindex(columns=TRADE_COLS).to_csv(FW3 / 'trades.csv', index=False)
reg7 = TMP / 'reg_real.jsonl'; shutil.copy(REG0, reg7)
t7 = []
for sd_ in EV.schedule(pd.Timestamp('2028-02-12')):
    e = EV.main(['calibrate', '--test-forward', str(FW3), '--test-registry', str(reg7), '--test-protocol', str(PD / 'G8_PROTOCOL_v2.json'),
                 '--test-manifest', str(PD / 'FREEZE_MANIFEST_v2.json'), '--test-now', f'{sd_}T08:00:00Z', '--quiet'])
    t7.append((sd_, e['type'], e['counts_toward_slot'], round(e['delta_per_side'], 6)))
o = run_status(FW3, reg7, ['--test-now', '2028-03-01T00:00:00Z'])
B['B7_real_calib_slot'] = dict(entries=t7, calibs_applied=len(o['calibs_applied']), states=(o['by_mode']['retro']['state'], o['by_mode']['forward']['state'], o['state']),
                               ok=[t[1] for t in t7] == ['REAL_CALIB', 'REAL_SLOT_LOCKED', 'REAL_SLOT_LOCKED', 'REAL_SLOT_LOCKED', 'REAL_CALIB'] and len(o['calibs_applied']) == 2)
# B8 生效錨定月規則與 git 錨定
cases = {'2026-11-10T05:29:00Z': '2026-11', '2026-11-10T05:31:00Z': '2026-12', '2026-12-09T00:00:00Z': '2026-12', '2026-12-10T06:00:00Z': '2027-01'}
got = {k: EV.effective_first_anchor(k) for k in cases}
ga = EV.git_anchor(TMP / 'reg0.jsonl')
B['B8_effective_anchor'] = dict(got=got, expect=cases, git_uncommitted=ga, ok=(got == cases and not ga.get('ok')))
# B9 正式呼叫（本資料夾檔案）：凍結前應 INVALID（MANIFEST 缺）；凍結後未錨定前應 INVALID（ANCHOR_*）
o = EV.main(['status', '--official'])
o.setdefault('ledger_sha256', None)
B['B9_official_call_now'] = dict(label=o['label'], OFFICIAL=o['OFFICIAL'], problems=o['problems'], state_if_valid=o['state_if_valid'],
                                 ledger_sha256=o['ledger_sha256'], note='凍結後、外部錨定（BACKLOG 行＋git commit）完成前，正式呼叫必須 INVALID')
B_ok = all(v.get('ok', True) for k, v in B.items() if k != 'B9_official_call_now')
print('B', json.dumps({k: v.get('ok') for k, v in B.items()}, ensure_ascii=False))
print(json.dumps(B['B6_P9_no_real_fills']['entries'], ensure_ascii=False)); print(json.dumps(B['B7_real_calib_slot'], ensure_ascii=False)); print(json.dumps(B['B9_official_call_now'], ensure_ascii=False)[:800])
res = dict(utc=L.now_utc(), protocol_dir=str(PD), protocol_sha256=L.sha(PD / 'G8_PROTOCOL_v2.json'), evaluator_sha256=L.sha(HERE / 'g8_evaluate_v2.py'),
           A=A_res, B=B, all_ok=bool(A_res['n_mismatch'] == 0 and B_ok))
if A.record:
    L.save_key('EVALUATOR_TEST', res, need=('G8_FREEZE',))
shutil.rmtree(TMP, ignore_errors=True)
print('ALL_OK', res['all_ok'])
