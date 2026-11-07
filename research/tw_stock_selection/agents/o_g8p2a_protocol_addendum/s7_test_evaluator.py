#!/usr/bin/env python3
"""步驟 7：判定器測試（PREREG 之後；合成帳本與暫存 git repo 全部在 scratchpad；不碰前瞻資料夾）。
 A  一致性：≥ 300 條合成帳本（24／36 個有效月、作廢、作廢延長）：判定器 vs 純量參考實作（refimpl）vs OC 向量化程式（g8alib.combined）→ 0 不一致；
    逐路徑子集：判定器合併狀態 vs C1（G8P-2 判定器原樣）→ 0 違反
 C  逐月正式呼叫模擬（30 條 × 26 個月，每月寫鏈）：最終狀態與參考一致、沒有 RETRO_CHANGE、鏈可驗證
 B  治理攻擊：回溯更正、open 月份插回（期限內／逾期）、adj 覆寫、S1 無 REAL_CALIB、V3_PRESENT、登錄簿竄改／逾期／重複、鏈竄改／缺檔／倒退、
    遠端：正常、committer date 偽造（push 晚於期限）、未推送、BACKLOG 未推送、遠端鏈截尾、遠端不可達、見證 push 早／晚於期限、生效月延後
用法：python3 -I -B s7_test_evaluator.py"""
import sys, json, math, shutil, subprocess, importlib.util, copy, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import g8alib as L
import synth as SY
import refimpl as R

L.check_reg('PREREG', files=False)
spec = importlib.util.spec_from_file_location('g8ev2a_t', HERE / 'g8_evaluate_v2a.py'); EV = importlib.util.module_from_spec(spec); spec.loader.exec_module(EV)
EV2 = EV.load_v2()
PROTO = HERE / 'G8_PROTOCOL_v2a.json'; PA = json.loads(PROTO.read_text(encoding='utf-8'))
PAR = dict(sigma_design=L.SIGMA_D, c=L.C_INFL, c_d=L.CD_V2, b_v2=L.B_V2, b_a=PA['params']['b_a'])
SEL = PA['params']['selected']
TMP = L.SCR / 'evtest'; shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')


def genesis():
    g = dict(seq=0, type='GENESIS', utc='2026-11-07T00:00:00+00:00', prev_sha=EV.ZERO, protocol_sha256=L.sha(PROTO), evaluator_sha256=L.sha(HERE / 'g8_evaluate_v2a.py'),
             note='測試用 GENESIS')
    g['entry_sha'] = EV.entry_sha(g); return g


G0 = genesis()
MAN = TMP / 'man.json'
MAN.write_text(json.dumps(dict(protocol_sha256=L.sha(PROTO), evaluator_sha256=L.sha(HERE / 'g8_evaluate_v2a.py'), status_chain_genesis_entry_sha=G0['entry_sha'])))


def chain_file(p, extra=()):
    lines = [EV.canon(G0)]; prev = G0['entry_sha']
    for i, e in enumerate(extra, start=1):
        e = dict(e, seq=i, prev_sha=prev); e.pop('entry_sha', None); e['entry_sha'] = EV.entry_sha(e); prev = e['entry_sha']; lines.append(EV.canon(e))
    Path(p).write_text('\n'.join(lines) + '\n', encoding='utf-8'); return p


def reg_entry(prev, seq, sd, typ='NO_REAL_FILLS', delta=0.0, utc=None, vf=None):
    e = dict(seq=seq, type=typ, sched_date=sd, utc=utc or f'{(pd.Timestamp(sd) + pd.Timedelta(days=1)).date()}T08:00:00+00:00',
             valid_from=vf or f'{(pd.Timestamp(sd) + pd.Timedelta(days=1)).date()}T08:00:00+00:00', counts_toward_slot=(typ == 'REAL_CALIB'),
             applied=(typ == 'REAL_CALIB'), delta_per_side=float(delta), prev_sha=prev)
    e['entry_sha'] = EV2.entry_sha(e); return e


def registry(p, until, real=None):
    """GENESIS（G8P-2 原檔第 1 行）＋到 until 的每個排定日一筆；real：{sched_date: (Δ, valid_from)} → REAL_CALIB。"""
    g = (L.V2_DIR / 'calib_registry.jsonl').read_text(encoding='utf-8').splitlines()[0]
    lines = [g]; prev = json.loads(g)['entry_sha']
    for i, sd in enumerate(EV2.schedule(pd.Timestamp(until)), start=1):
        if real and sd in real: e = reg_entry(prev, i, sd, 'REAL_CALIB', real[sd][0], vf=real[sd][1])
        else: e = reg_entry(prev, i, sd)
        lines.append(EV2.canon(e)); prev = e['entry_sha']
    Path(p).write_text('\n'.join(lines) + '\n', encoding='utf-8'); return p


def run(fwd, reg, chain, now, extra=(), write=False, man=MAN, proto=PROTO):
    a = ['status', '--test-protocol', str(proto), '--test-manifest', str(man), '--test-chain', str(chain), '--test-registry', str(reg),
         '--test-forward', str(fwd), '--test-now', now, '--quiet'] + list(extra)
    if '--test-repo' not in extra: a.append('--test-skip-remote')
    if write: a.append('--test-write-chain')
    return EV.main(a)


def run_v2(fwd, reg, now):
    return EV2.main(['status', '--test-forward', str(fwd), '--test-registry', str(reg), '--test-protocol', str(L.V2_DIR / 'G8_PROTOCOL_v2.json'),
                     '--test-manifest', str(L.V2_DIR / 'FREEZE_MANIFEST_v2.json'), '--test-skip-anchor', '--test-now', now, '--quiet'])


def items_of(expect):
    return [(k, x) for _, k, x in expect if k in ('valid', 'void')]


def oc_state(xs24):
    Z = L.all_stats(np.asarray(xs24, float)[None, :])
    o = L.combined(Z, SEL, PAR['b_a'])
    return 'S1_EXCEPTION_EVIDENCE' if o['up'][0] else ('FORWARD_FAILED' if o['fail'][0] else 'FORWARD_VALIDATING_NO_REJECTION_24'), o


def witness_entry(expect, utc='2026-12-01T00:00:00+00:00'):
    months = {an: (['valid', x, None] if k == 'valid' else ['void', None, 'V0-V2']) for an, k, x in expect}
    return dict(type='STATUS', utc=utc, months=months, state='FORWARD_VALIDATING', literal=None, note='測試用見證（全部月份）')


def last_now(expect):
    return (EV.month_deadline(expect[-1][0]) - pd.Timedelta(days=1)).isoformat()


rng = np.random.default_rng(20261107)
out = {}

# ---------------------------------------------------------------- A 一致性
REG_BIG = registry(TMP / 'reg_big.jsonl', '2030-12-31')
A_rows, A_mis, A_sub = [], [], []
NA = 300
for i in range(NA):
    mu = [-1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 5.0][i % 7]; sg = [6.14, 8.4, 10.4][(i // 7) % 3]
    long_ = i >= 250
    tv = 36 if long_ else 24
    maxv12 = 4 if (long_ and i % 2 == 0) else 2
    months, per, nval, nv12 = [], pd.Period('2026-11', 'M'), 0, 0
    while nval < tv:
        vm = rng.random() < (0.15 if long_ else 0.06) and (nval >= 12 or nv12 < maxv12)
        if vm and nval < 12: nv12 += 1
        months.append(dict(anchor=str(per), x=float(mu + 0.5 + sg * rng.standard_normal()), void=bool(vm))); per += 1
        if not vm: nval += 1
    fwd = TMP / 'fwdA'
    exp = SY.make_ledger(fwd, months, rng, jitter=1.0)
    ch = chain_file(TMP / 'chainA.jsonl', [witness_entry(exp)])
    o = run(fwd, REG_BIG, ch, last_now(exp))
    its = items_of(exp)
    ref, s2, sn, lit = R.combined_state(its, PAR, SEL)
    ok = (not o['problems'] and o['state_if_valid'] == ref and 'RETRO_CHANGE' not in o['flags'])
    # C1（v2 原樣）vs 參考 v2 規則
    v2ref, _ = R.seq_rule(its, R.rules(PAR, SEL)[0])
    c1 = [o['by_mode'][m]['C1_v2_as_implemented']['state'] for m in ('retro', 'forward')]
    ok = ok and all(c == v2ref for c in c1)
    # OC 向量化（前 24 個有效月、前 12 月作廢 ≤ 2）
    nvb = 0; cnt = 0; first24 = []
    for k, x in its:
        if k == 'void' and cnt < 12: nvb += 1
        if k == 'valid': cnt += 1; first24.append(x) if len(first24) < 24 else None
    ocs = None
    if nvb <= 2 and len(first24) == 24:
        ocs, _ = oc_state(first24)
        trunc = []; c = 0
        for k, x in its:
            if c == 24: break
            trunc.append((k, x)); c += (k == 'valid')
        ref24, _, _, _ = R.combined_state(trunc, PAR, SEL)
        ok = ok and (ocs == ref24)
    # 逐路徑子集：合併 vs C1
    fin = o['state_if_valid']
    sub_bad = (fin == 'S1_EXCEPTION_EVIDENCE' and any(c != 'S1_EXCEPTION_EVIDENCE' for c in c1)) or (any(c == 'FORWARD_FAILED' for c in c1) and fin != 'FORWARD_FAILED')
    A_rows.append(dict(i=i, n_valid=cnt, nvb12=nvb, evaluator=fin, ref=ref, v2=v2ref, oc24=ocs, ok=bool(ok), sub_bad=bool(sub_bad)))
    if not ok: A_mis.append(A_rows[-1] | dict(problems=o['problems'], flags=o['flags']))
    if sub_bad: A_sub.append(A_rows[-1])
D = pd.DataFrame(A_rows)
out['A_consistency'] = dict(n_paths=NA, n_mismatch=len(A_mis), mismatches=A_mis[:5], n_pathwise_violations=len(A_sub),
                            n_36=int((D.n_valid >= 36).sum()), n_ext=int((D.nvb12 > 2).sum()), n_oc_compared=int(D.oc24.notna().sum()),
                            states=D.evaluator.value_counts().to_dict(), v2_states=D.v2.value_counts().to_dict(),
                            n_v2a_stricter_than_v2=int(sum(R.ORDER.index(a) < R.ORDER.index(b) for a, b in zip(D.evaluator, D.v2))))
print('A', {k: v for k, v in out['A_consistency'].items() if k != 'mismatches'}, A_mis[:2])

# ---------------------------------------------------------------- C 逐月正式呼叫模擬
C_rows = []
for i in range(30):
    mu = [-1.0, 0.5, 2.0, 5.0, 8.0][i % 5]; sg = [6.14, 9.0][i % 2]
    xs = [float(mu + 0.5 + sg * rng.standard_normal()) for _ in range(26)]
    ch = chain_file(TMP / 'chainC.jsonl')
    fwd = TMP / 'fwdC'; flags_all = []; states = []
    for k in range(1, 27):
        exp = SY.make_ledger(fwd, SY.months_seq(xs[:k]), np.random.default_rng(i * 100 + 7), jitter=0.0)
        now = (pd.Period(exp[-1][0], 'M') + 1).to_timestamp() + pd.Timedelta(days=27)
        o = run(fwd, REG_BIG, ch, now.isoformat() + '+00:00', write=True)
        flags_all += [f for f in o['flags'] if f != 'REMOTE_SKIPPED_TEST']; states.append(o['state_if_valid'])
        assert not o['problems'], o['problems']
    ref, _, _, _ = R.combined_state(items_of(exp), PAR, SEL)
    ents, cp = EV.read_chain(ch, G0['entry_sha'])
    C_rows.append(dict(i=i, final=states[-1], ref=ref, ok=(states[-1] == ref and not any(f.startswith('RETRO') for f in flags_all) and not cp and len(ents) == 27),
                       flags=sorted(set(flags_all)), chain_n=len(ents)))
out['C_monthly_sim'] = dict(n_paths=30, n_calls=30 * 26, n_mismatch=int(sum(not r['ok'] for r in C_rows)), bad=[r for r in C_rows if not r['ok']][:3],
                            states=pd.Series([r['final'] for r in C_rows]).value_counts().to_dict())
print('C', out['C_monthly_sim'])

# ---------------------------------------------------------------- B 治理
B = {}
REG_S = registry(TMP / 'reg_s.jsonl', '2028-12-31')
def sim_monthly(xs, fwd, ch, opens=(), upto=None, now_fn=None, reg=REG_S):
    """逐月呼叫並寫鏈，回傳最後一次輸出。"""
    o = None
    for k in range(1, (upto or len(xs)) + 1):
        exp = SY.make_ledger(fwd, SY.months_seq(xs[:k], opens=tuple(j for j in opens if j < k)), np.random.default_rng(1), jitter=0.0)
        now = (pd.Period(exp[-1][0], 'M') + 1).to_timestamp() + pd.Timedelta(days=27)
        o = run(fwd, reg, ch, now.isoformat() + '+00:00', write=True)
    return o, now
# B1 回溯更正
xs = [0.3, -1.0, 0.6, 0.2, -0.4, 0.8, 0.1, 0.5, 0.3, 0.2, 0.4, 0.2]          # 判定線＝x−0.5 → Σ＝−2.8 → 字面 G8 失敗
fwd = TMP / 'fwdB1'; ch = chain_file(TMP / 'chainB1.jsonl')
o1, now1 = sim_monthly(xs, fwd, ch)
v2_before = run_v2(fwd, REG_S, now1.isoformat() + '+00:00')['state_if_valid']
xs2 = list(xs); xs2[1] = 3.0
SY.make_ledger(fwd, SY.months_seq(xs2), np.random.default_rng(1), jitter=0.0)
now2 = (now1 + pd.Timedelta(days=5)).isoformat() + '+00:00'
o2 = run(fwd, REG_S, ch, now2, write=True)
v2_after = run_v2(fwd, REG_S, now2)['state_if_valid']
B['B1_retro_correction'] = dict(v2_before=v2_before, v2_after=v2_after, v2_changed=v2_before != v2_after, v2a_before=o1['state_if_valid'], v2a_after=o2['state_if_valid'],
                                v2a_changed=o1['state_if_valid'] != o2['state_if_valid'], v2a_flags=o2['flags'], retro=o2['retro_changes'], literal=o2['literal_g8'],
                                ok=(o2['state_if_valid'] == 'FORWARD_FAILED' and 'RETRO_CHANGE' in o2['flags'] and o2['literal_g8'] == 'FAIL'))
# B2a open 月份插回（期限內）：第 11 個月很差且 open；第 12、13 個月完成時 v2 以 1～10、12、13 判字面 G8（通過）；期限內補完插回 → v2 翻成失敗。
# v2a：open 期間 C2／C3 延後，字面 G8 不發出；插回後才判 → 已發出的判定 0 次翻轉
fwd = TMP / 'fwdB2'
months = SY.months_seq([0.9] * 13); months[10]['x'] = -40.0
REG_T = registry(TMP / 'reg_t.jsonl', '2029-12-31')
exp = SY.make_ledger(fwd, [dict(m, open=(j == 10)) for j, m in enumerate(months)], np.random.default_rng(1))
ch = chain_file(TMP / 'chainB2.jsonl', [witness_entry([e for e in exp if e[1] != 'open'], utc='2026-12-01T00:00:00+00:00')])
nowB = '2027-12-20T00:00:00+00:00'
assert pd.Timestamp(nowB) < EV.month_deadline(months[10]['anchor'])
v2_t1 = run_v2(fwd, REG_T, nowB); a_t1 = run(fwd, REG_T, ch, nowB, write=True)
SY.make_ledger(fwd, months, np.random.default_rng(1))
nowB2 = '2027-12-24T00:00:00+00:00'
v2_t2 = run_v2(fwd, REG_T, nowB2); a_t2 = run(fwd, REG_T, ch, nowB2, write=True)
def lit_v2(o): g = o['by_mode']['retro']['g8_literal']; return None if not g else ('PASS' if g.get('passed') else 'FAIL')
B['B2a_open_insert_within_deadline'] = dict(
    v2=dict(t1=v2_t1['state_if_valid'], t1_lit=lit_v2(v2_t1), t2=v2_t2['state_if_valid'], t2_lit=lit_v2(v2_t2)),
    v2a=dict(t1=a_t1['state_if_valid'], t1_lit=a_t1['literal_g8'], t1_deferred=a_t1['by_mode']['retro']['C3_new_rule']['deferred_at'], t2=a_t2['state_if_valid'], t2_lit=a_t2['literal_g8'],
             t2_flags=a_t2['flags']),
    v2_issued_decision_reversed=bool(lit_v2(v2_t1) == 'PASS' and lit_v2(v2_t2) == 'FAIL'),
    v2a_issued_decision_reversed=bool(a_t1['literal_g8'] is not None and a_t1['literal_g8'] != a_t2['literal_g8']),
    ok=bool(a_t1['literal_g8'] is None and a_t2['literal_g8'] == 'FAIL' and 'RETRO_CHANGE' not in a_t2['flags']))
# B2b open 月份逾期 → V3、不得插回
fwd = TMP / 'fwdB2b'
months = SY.months_seq([0.6] * 14); months[1]['x'] = -40.0
exp = SY.make_ledger(fwd, [dict(m, open=(j == 1)) for j, m in enumerate(months)], np.random.default_rng(1))
ch = chain_file(TMP / 'chainB2b.jsonl', [witness_entry([e for e in exp if e[1] != 'open'], utc='2026-12-01T00:00:00+00:00')])
nowC = (EV.month_deadline(months[13]['anchor']) - pd.Timedelta(days=1)).isoformat()
v2_c1 = run_v2(fwd, REG_T, nowC); a_c1 = run(fwd, REG_T, ch, nowC, write=True)
SY.make_ledger(fwd, months, np.random.default_rng(1))
nowC2 = (pd.Timestamp(nowC) + pd.Timedelta(days=3)).isoformat()
v2_c2 = run_v2(fwd, REG_T, nowC2); a_c2 = run(fwd, REG_T, ch, nowC2, write=True)
row1 = [r for r in a_c2['by_mode']['retro']['rows'] if r['anchor'] == months[1]['anchor']][0]
B['B2b_open_overdue_V3'] = dict(v2=dict(t1=v2_c1['state_if_valid'], t2=v2_c2['state_if_valid'], changed=v2_c1['state_if_valid'] != v2_c2['state_if_valid']),
                                v2a=dict(t1=a_c1['state_if_valid'], t2=a_c2['state_if_valid'], changed=a_c1['state_if_valid'] != a_c2['state_if_valid']),
                                month2_after_insert=row1,
                                v2a_honest=None,
                                ok=(row1.get('void_kind') == 'V3' and R.ORDER.index(a_c2['state_if_valid']) <= R.ORDER.index(a_c1['state_if_valid'])))
SY.make_ledger(fwd, months, np.random.default_rng(1)); chh = chain_file(TMP / 'chainB2h.jsonl', [witness_entry(SY.make_ledger(fwd, months, np.random.default_rng(1)))])
B['B2b_open_overdue_V3']['v2a_honest'] = run(fwd, REG_T, chh, nowC)['state_if_valid']
B['B2b_open_overdue_V3']['v2_honest'] = run_v2(fwd, REG_T, nowC)['state_if_valid']
B['B2b_open_overdue_V3']['permanent_open_gain_v2'] = R.ORDER.index(v2_c1['state_if_valid']) > R.ORDER.index(B['B2b_open_overdue_V3']['v2_honest'])
B['B2b_open_overdue_V3']['permanent_open_gain_v2a'] = R.ORDER.index(a_c1['state_if_valid']) > R.ORDER.index(B['B2b_open_overdue_V3']['v2a_honest'])
B['B2b_open_overdue_V3']['insert_after_deadline_gain_v2a'] = R.ORDER.index(a_c2['state_if_valid']) > R.ORDER.index(a_c1['state_if_valid'])
# B3 adj 最大值
fwd = TMP / 'fwdB3'
SY.make_ledger(fwd, SY.months_seq([2.0] * 3), turnover=0.8)
fr = pd.read_csv(fwd / 'forward_returns.csv'); fr['excess_stress_pp'] = 9.0; fr.to_csv(fwd / 'forward_returns.csv', index=False)
REG_R = registry(TMP / 'reg_r.jsonl', '2028-03-01', real={'2027-02-12': (0.004, '2026-11-01T00:00:00+00:00'), '2028-02-12': (0.001, '2026-11-01T00:00:00+00:00')})
ch = chain_file(TMP / 'chainB3.jsonl')
o = run(fwd, REG_R, ch, '2027-01-20T00:00:00+00:00')
xa = [r.get('x') for r in o['by_mode']['retro']['rows']]; xc1 = [r.get('x') for r in o['by_mode']['retro']['C1_v2_as_implemented'].get('anchors', [])] if 'anchors' in o['by_mode']['retro']['C1_v2_as_implemented'] else None
B['B3_adj_max'] = dict(x_v2a_retro=xa, expect_month1=2.0 - 100 * 0.004 * 0.8, ok=abs(xa[0] - (2.0 - 100 * 0.004 * 0.8)) < 1e-9, problems=o['problems'])
# B4 S1 無／有 REAL_CALIB；V3_PRESENT
fwd = TMP / 'fwdB4'
exp = SY.make_ledger(fwd, SY.months_seq([9.0] * 24), np.random.default_rng(2), jitter=0.5)
ch = chain_file(TMP / 'chainB4.jsonl', [witness_entry(exp)])
o = run(fwd, REG_T, ch, last_now(exp))
B['B4a_S1_no_real_calib'] = dict(state=o['state_if_valid'], flags=o['flags'], text=o['state_text'], ok=(o['state_if_valid'] == 'S1_EXCEPTION_EVIDENCE' and 'NO_REAL_CALIB' in o['flags'] and '不得引用' in (o['state_text'] or '')))
REG_R2 = registry(TMP / 'reg_r2.jsonl', '2029-12-31', real={'2027-02-12': (0.0001, '2027-02-13T08:00:00+00:00')})
o = run(fwd, REG_R2, ch, last_now(exp))
B['B4b_S1_with_real_calib'] = dict(state=o['state_if_valid'], flags=o['flags'], shas=o['s1_real_calib_entry_shas'], text=o['state_text'],
                                   ok=(o['state_if_valid'] == 'S1_EXCEPTION_EVIDENCE' and 'NO_REAL_CALIB' not in o['flags'] and bool(o['s1_real_calib_entry_shas'])))
months = SY.months_seq([9.0] * 25);
exp = SY.make_ledger(fwd, [dict(m, open=(j == 3)) for j, m in enumerate(months)], np.random.default_rng(2), jitter=0.5)
ch = chain_file(TMP / 'chainB4c.jsonl', [witness_entry([e for e in exp if e[1] != 'open'])])
o = run(fwd, REG_R2, ch, last_now(exp))
B['B4c_V3_present'] = dict(state=o['state_if_valid'], flags=o['flags'], text=o['state_text'], ok=(o['state_if_valid'] == 'S1_EXCEPTION_EVIDENCE' and 'V3_PRESENT' in o['flags']))
# B5 登錄簿
fwd = TMP / 'fwdB5'; SY.make_ledger(fwd, SY.months_seq([1.0] * 3)); ch = chain_file(TMP / 'chainB5.jsonl')
lines = Path(REG_S).read_text().splitlines()
t = TMP / 'reg_tamper.jsonl'; e = json.loads(lines[1]); e['delta_per_side'] = 0.5; t.write_text('\n'.join([lines[0], EV2.canon(e)] + lines[2:]) + '\n')
o = run(fwd, t, ch, '2027-04-01T00:00:00+00:00'); B['B5a_registry_tamper'] = dict(problems=o['problems'], ok=any('REGISTRY_ENTRY_SHA' in p for p in o['problems']) and o['state'] is None)
g = lines[0]; prev = json.loads(g)['entry_sha']; e1 = reg_entry(prev, 1, '2027-02-12', utc='2027-03-20T00:00:00+00:00')
t = TMP / 'reg_late.jsonl'; t.write_text(g + '\n' + EV2.canon(e1) + '\n')
o = run(fwd, t, ch, '2027-04-01T00:00:00+00:00'); B['B5b_registry_late'] = dict(problems=o['problems'], ok=any('REGISTRY_LATE' in p for p in o['problems']))
o_v2 = run_v2(fwd, t, '2027-04-01T00:00:00+00:00'); B['B5b_registry_late']['v2_problems'] = o_v2['problems']
e1 = reg_entry(prev, 1, '2027-02-12'); e2 = reg_entry(e1['entry_sha'], 2, '2027-02-12')
t = TMP / 'reg_dup.jsonl'; t.write_text(g + '\n' + EV2.canon(e1) + '\n' + EV2.canon(e2) + '\n')
o = run(fwd, t, ch, '2027-04-01T00:00:00+00:00'); B['B5c_registry_duplicate'] = dict(problems=o['problems'], ok=any('REGISTRY_DUPLICATE' in p for p in o['problems']))
# B6 鏈
ch = chain_file(TMP / 'chainB6.jsonl', [dict(type='STATUS', utc='2027-01-01T00:00:00+00:00', months={}, state='FORWARD_VALIDATING', literal=None),
                                         dict(type='STATUS', utc='2027-02-01T00:00:00+00:00', months={}, state='FORWARD_VALIDATING', literal=None)])
ls = ch.read_text().splitlines(); e = json.loads(ls[1]); e['state'] = 'FORWARD_FAILED'; tch = TMP / 'chain_t.jsonl'; tch.write_text('\n'.join([ls[0], EV.canon(e), ls[2]]) + '\n')
o = run(fwd, REG_S, tch, '2027-02-20T00:00:00+00:00'); B['B6a_chain_tamper'] = dict(problems=o['problems'], ok=any('CHAIN_ENTRY_SHA' in p for p in o['problems']) and o['state'] is None)
o = run(fwd, REG_S, TMP / 'nochain.jsonl', '2027-02-20T00:00:00+00:00'); B['B6b_chain_missing'] = dict(problems=o['problems'], ok='CHAIN_MISSING' in o['problems'])
ch2 = chain_file(TMP / 'chainB6c.jsonl', [dict(type='STATUS', utc='2027-02-01T00:00:00+00:00', months={}, state='FORWARD_VALIDATING', literal=None),
                                          dict(type='STATUS', utc='2027-01-01T00:00:00+00:00', months={}, state='FORWARD_VALIDATING', literal=None)])
o = run(fwd, REG_S, ch2, '2027-02-20T00:00:00+00:00'); B['B6c_chain_utc_backwards'] = dict(problems=o['problems'], ok=any('UTC_BACKWARDS' in p for p in o['problems']))
# B7 CLI／協定
o = EV.main(['status', '--official', '--quiet', '--test-protocol', str(PROTO), '--test-manifest', str(MAN), '--test-chain', str(ch), '--test-registry', str(REG_S),
             '--test-forward', str(fwd), '--test-skip-remote', '--test-now', '2027-01-01T00:00:00Z'])
B['B7a_extra_flags_non_official'] = dict(label=o['label'], ok=(o['OFFICIAL'] is False and o['label'].startswith('NON_OFFICIAL')))
pt = TMP / 'proto_t.json'; pj = copy.deepcopy(PA); pj['params']['b_a'] = 1.0; pt.write_text(json.dumps(pj, ensure_ascii=False))
o = run(fwd, REG_S, ch, '2027-01-01T00:00:00Z', proto=pt); B['B7b_protocol_tamper'] = dict(problems=o['problems'], ok='PROTOCOL_SHA_MISMATCH' in o['problems'] and o['state'] is None)
# B8 生效月延後：C2／C3 在 active_from 之前不作用
fwd = TMP / 'fwdB8'; exp = SY.make_ledger(fwd, SY.months_seq(xs)); ch = chain_file(TMP / 'chainB8.jsonl', [witness_entry(exp)])
o = run(fwd, REG_S, ch, last_now(exp), extra=['--test-first-anchor', '2028-01'])
B['B8_late_effective'] = dict(state=o['state_if_valid'], components=o['components'], ok=(o['components']['retro'][1:] == ['FORWARD_VALIDATING', 'FORWARD_VALIDATING'] and o['state_if_valid'] == 'FORWARD_FAILED'))

# ---------------------------------------------------------------- 遠端（暫存 git repo）
def git(d, *a, env=None, check=True):
    r = subprocess.run(['git', '-C', str(d)] + list(a), capture_output=True, text=True, env=env or GIT_ENV)
    if check and r.returncode: raise RuntimeError(r.stderr)
    return r.stdout.strip()


def mkrepo(name):
    root = TMP / name; shutil.rmtree(root, ignore_errors=True)
    W, O = root / 'w', root / 'o.git'; W.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', '--bare', str(O)], check=True)
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(W)], check=True)
    git(W, 'remote', 'add', 'origin', str(O))
    ag = W / 'research/tw_stock_selection/agents'; (ag / 'o_g8p2a_protocol_addendum').mkdir(parents=True); (ag / 'o_g8p2_protocol_refreeze').mkdir(parents=True)
    shutil.copy(L.V2_DIR / 'FREEZE_MANIFEST_v2.json', ag / 'o_g8p2_protocol_refreeze' / 'FREEZE_MANIFEST_v2.json')
    (ag / 'BACKLOG.md').write_text('# BACKLOG\n')
    return dict(W=W, O=O, ag=ag, man=ag / 'o_g8p2a_protocol_addendum/FREEZE_MANIFEST_v2a.json', chain=ag / 'o_g8p2a_protocol_addendum/status_chain.jsonl',
                backlog=ag / 'BACKLOG.md', v2man=ag / 'o_g8p2_protocol_refreeze/FREEZE_MANIFEST_v2.json', events=root / 'events.json', ev=[])


def commit(r, msg, cdate=None, paths=('.',)):
    env = dict(GIT_ENV);
    if cdate: env.update(GIT_COMMITTER_DATE=cdate, GIT_AUTHOR_DATE=cdate)
    git(r['W'], 'add', *paths); git(r['W'], 'commit', '-q', '-m', msg, env=env)
    return git(r['W'], 'rev-parse', 'HEAD')


def push(r, ts):
    git(r['W'], 'push', '-q', 'origin', 'main')
    r['ev'].append(dict(timestamp=ts, after=git(r['W'], 'rev-parse', 'HEAD'), ref='refs/heads/main'))
    r['events'].write_text(json.dumps(r['ev']))


def rrun(r, fwd, reg, now, extra=()):
    return run(fwd, reg, r['chain'], now, man=r['man'],
               extra=['--test-repo', str(r['W']), '--test-push-events', str(r['events']), '--test-backlog', str(r['backlog']), '--test-v2-manifest', str(r['v2man'])] + list(extra))


def setup(r, cdate, push_ts, backlog_push=True):
    shutil.copy(MAN, r['man']); chain_file(r['chain'])
    commit(r, 'v2 manifest', '2026-11-05T23:40:00+00:00', paths=(str(r['v2man']),)); push(r, '2026-11-06T00:04:47Z')
    commit(r, 'v2a freeze', cdate)
    if push_ts: push(r, push_ts)
    with open(r['backlog'], 'a') as fh: fh.write(f"- [G8P-2a 凍結錨定] FREEZE_MANIFEST_v2a.json sha256 {L.sha(r['man'])}\n")
    commit(r, 'backlog anchor', cdate)
    if push_ts and backlog_push: push(r, push_ts)


fwd = TMP / 'fwdR'; SY.make_ledger(fwd, SY.months_seq([1.0] * 2))
NOWR = '2027-01-20T00:00:00+00:00'
r = mkrepo('r1'); setup(r, '2026-11-09T10:00:00+00:00', '2026-11-09T12:00:00Z')
o = rrun(r, fwd, REG_S, NOWR)
B['R1_normal'] = dict(problems=o['problems'], flags=o['flags'], eff=o['effective_first_anchor_v2a'], eff_v2=o['effective_first_anchor_v2'],
                      push=o['remote_anchor'].get('manifest_push'), ok=(not o['problems'] and o['effective_first_anchor_v2a'] == '2026-11' and o['effective_first_anchor_v2'] == '2026-11'
                                                                       and not any(f.startswith('REMOTE_UNVERIFIED') for f in o['flags'])))
r = mkrepo('r2'); setup(r, '2026-11-08T00:00:00+00:00', '2026-11-12T03:00:00Z')
o = rrun(r, fwd, REG_S, NOWR)
B['R2_forged_committer_date'] = dict(local_cI=o['remote_anchor'].get('manifest_first_commit_local_cI'), push=o['remote_anchor'].get('manifest_push'), flags=o['flags'],
                                     eff=o['effective_first_anchor_v2a'], ok=('ANCHOR_AFTER_DEADLINE' in o['flags'] and o['effective_first_anchor_v2a'] != '2026-11'))
r = mkrepo('r3'); setup(r, '2026-11-09T10:00:00+00:00', None)
o = rrun(r, fwd, REG_S, NOWR)
B['R3_not_pushed'] = dict(problems=o['problems'], ok=('ANCHOR_MANIFEST_NOT_ON_ORIGIN' in o['problems'] and o['state'] is None))
r = mkrepo('r4'); setup(r, '2026-11-09T10:00:00+00:00', '2026-11-09T12:00:00Z', backlog_push=False)
o = rrun(r, fwd, REG_S, NOWR)
B['R4_backlog_not_pushed'] = dict(problems=o['problems'], ok=('ANCHOR_BACKLOG_REMOTE_MISSING' in o['problems']))
# R5 遠端鏈截尾
r = mkrepo('r5'); setup(r, '2026-11-09T10:00:00+00:00', '2026-11-09T12:00:00Z')
for k in (1, 2):
    SY.make_ledger(fwd, SY.months_seq([1.0] * k)); rrun(r, fwd, REG_S, f'2027-0{k}-20T00:00:00+00:00', extra=['--test-write-chain'])
commit(r, 'chain'); push(r, '2027-02-21T00:00:00Z')
lines = r['chain'].read_text().splitlines(); r['chain'].write_text('\n'.join(lines[:-1]) + '\n')
o = rrun(r, fwd, REG_S, '2027-02-22T00:00:00+00:00')
B['R5_chain_truncated_vs_remote'] = dict(problems=o['problems'], ok=('CHAIN_NOT_EXTENSION_OF_REMOTE' in o['problems']))
# R6 遠端不可達
r = mkrepo('r6'); setup(r, '2026-11-09T10:00:00+00:00', '2026-11-09T12:00:00Z'); git(r['W'], 'remote', 'set-url', 'origin', str(TMP / 'nonexistent.git'))
o = rrun(r, fwd, REG_S, NOWR)
B['R6_remote_unreachable'] = dict(flags=o['flags'], label=o['label'], ok=(any(f.startswith('REMOTE_UNVERIFIED') for f in o['flags']) and not o['OFFICIAL']))
# R7 見證：鏈條目在期限前／後 push
def witness_case(name, push_ts):
    r = mkrepo(name); setup(r, '2026-11-09T10:00:00+00:00', '2026-11-09T12:00:00Z')
    SY.make_ledger(fwd, SY.months_seq([1.0] * 1)); rrun(r, fwd, REG_S, '2026-12-28T00:00:00+00:00', extra=['--test-write-chain'])
    commit(r, 'chain witness'); push(r, push_ts)
    o = rrun(r, fwd, REG_S, '2027-03-05T00:00:00+00:00')
    return o, [x for x in o['by_mode']['retro']['rows'] if x['anchor'] == '2026-11'][0]
o, row = witness_case('r7a', '2026-12-29T00:00:00Z')
B['R7a_witness_pushed_before_deadline'] = dict(row=row, problems=o['problems'], ok=(row['state'] == 'valid' and not o['problems']))
o, row = witness_case('r7b', '2027-03-01T00:00:00Z')
B['R7b_witness_pushed_after_deadline'] = dict(row=row, deadline=str(EV.month_deadline('2026-11')), problems=o['problems'], ok=(row.get('void_kind') == 'V3'))
out['B_governance'] = B
n_ok = sum(1 for v in B.values() if v.get('ok')); n_b = len([v for v in B.values() if 'ok' in v])
b2b = B['B2b_open_overdue_V3']
attacks = dict(retro_correction=dict(v2_changed=B['B1_retro_correction']['v2_changed'], v2a_changed=B['B1_retro_correction']['v2a_changed']),
               open_insert_within_deadline=dict(v2_reversed=B['B2a_open_insert_within_deadline']['v2_issued_decision_reversed'],
                                                v2a_reversed=B['B2a_open_insert_within_deadline']['v2a_issued_decision_reversed']),
               open_insert_after_deadline=dict(v2_changed=b2b['v2']['changed'], v2a_favorable_change=b2b['insert_after_deadline_gain_v2a']),
               permanent_open_residual=dict(v2_gain=b2b['permanent_open_gain_v2'], v2a_gain=b2b['permanent_open_gain_v2a'],
                                            note='永久不補完：v2 不計作廢；v2a 計 V3 作廢、S1 加 V3_PRESENT 不得引用，但字面 G8 仍可能受益（K14）'))
out['attack_summary'] = dict(attacks,
    n_attacks_change_v2=int(attacks['retro_correction']['v2_changed']) + int(attacks['open_insert_within_deadline']['v2_reversed']) + int(attacks['open_insert_after_deadline']['v2_changed']),
    n_attacks_change_v2a=int(attacks['retro_correction']['v2a_changed']) + int(attacks['open_insert_within_deadline']['v2a_reversed']) + int(attacks['open_insert_after_deadline']['v2a_favorable_change']),
    n_residual_v2a=int(attacks['permanent_open_residual']['v2a_gain']))
n_synth = NA + 30 + 12
out['summary'] = dict(n_synthetic_ledger_paths=n_synth, n_evaluator_calls_approx=NA + 30 * 26 + 60, consistency_mismatch=out['A_consistency']['n_mismatch'] + out['C_monthly_sim']['n_mismatch'],
                      pathwise_violations=out['A_consistency']['n_pathwise_violations'], governance_tests_ok=f'{n_ok}/{n_b}',
                      evaluator_sha256=L.sha(HERE / 'g8_evaluate_v2a.py'), protocol_sha256=L.sha(PROTO), utc=L.now_utc())
print('B', json.dumps({k: v.get('ok') for k, v in B.items()}, ensure_ascii=False))
print('ATT', out['attack_summary'])
print('SUM', out['summary'])
(HERE / 'evaluator_tests.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
L.save_key('EVALUATOR_TEST', dict(file='evaluator_tests.json', sha256=L.sha(HERE / 'evaluator_tests.json'), summary=out['summary'], attack_summary=out['attack_summary']), need=('PREREG',))
shutil.rmtree(TMP, ignore_errors=True)
