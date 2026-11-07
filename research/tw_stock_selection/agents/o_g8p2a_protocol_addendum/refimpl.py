#!/usr/bin/env python3
"""G8P-2a 純量參考實作（依協定文字逐條、純 Python、不 import numpy 以外的研究程式）。
用於：煙霧測試（與 OC 向量化程式逐路徑比對）、判定器測試（與 g8_evaluate_v2a.py 比對）。

seq_rule(items, R)：items＝依錨定月順序的 ('valid', x)／('void', None)；R＝規則參數：
  denom ∈ {'v2'：max(c·σ_d, s_n)；'A'：c·max(σ_d, s_n)；'H'：max(c·σ_d, ŝ_HAC6,n)}、b、c_d、upgrade、sigcheck（A2）。
語意同 G8P-2 判定器 sequential()：第 12 個有效月先字面 G8（作廢 > 2 → 延長到 24 判一次；累計作廢 > 4 → 失敗）、再下修、再升級；
18 只下修；24 先下修再升級；之後每 6 個月只下修；先觸發者吸收。
combined_state(items, PAR, ver)：min_ORDER（v2 規則, 新規則）＝較不利者 ⇒ 升級需兩者都升級、失敗只要任一失敗。"""
import math

ORDER = ['FORWARD_FAILED', 'FORWARD_VALIDATING_NO_REJECTION_24', 'FORWARD_VALIDATING', 'UPGRADE_PENDING_24', 'S1_EXCEPTION_EVIDENCE']
SIG_MAX = 10.4


def _sd(v):
    n = len(v); m = sum(v) / n
    return math.sqrt(sum((a - m) ** 2 for a in v) / (n - 1))


def _hac(v, L=6):
    n = len(v); m = sum(v) / n; y = [a - m for a in v]
    s = sum(a * a for a in y) / n
    for k in range(1, min(L, n - 1) + 1):
        s += 2.0 * (1.0 - k / (L + 1.0)) * sum(y[t] * y[t - k] for t in range(k, n)) / n
    return math.sqrt(max(s, 1e-12))


def zstat(xs, R):
    n = len(xs); S = sum(xs); sn = _sd(xs)
    sd, c = R['sigma_design'], R['c']
    if R['denom'] == 'v2': sig = max(c * sd, sn)
    elif R['denom'] == 'A': sig = c * max(sd, sn)
    elif R['denom'] == 'H': sig = max(c * sd, _hac(xs, 6))
    else: raise ValueError(R['denom'])
    return S / (sig * math.sqrt(n))


def sig_ok(xs, R):
    if not R.get('sigcheck'): return True
    chk = [_sd(xs)]
    if len(xs) >= 12: chk.append(_sd(xs[-12:]))
    if len(xs) >= 24: chk.append(_sd(xs[-24:]))
    return max(chk) <= SIG_MAX + 1e-12


def seq_rule(items, R):
    xs = []; st = 'FORWARD_VALIDATING'; nvb = 0; ext = False; lit = None
    nv_all = sum(1 for k, _ in items if k == 'void')
    for k, x in items:
        if k == 'void':
            if len(xs) < 12: nvb += 1
            continue
        xs.append(x); n = len(xs)
        if st in ('FORWARD_FAILED', 'S1_EXCEPTION_EVIDENCE'): continue
        if n == 12:
            if nvb > R['max_void_12']: ext = True; lit = 'UNDECIDABLE_EXTENDED'
            elif sum(xs) <= 0: st = 'FORWARD_FAILED'; lit = 'FAIL'; continue
            else: lit = 'PASS'
        if n == 24 and ext:
            if sum(xs) <= 0 or nv_all > 2 * R['max_void_12']: st = 'FORWARD_FAILED'; lit = 'FAIL'; continue
            lit = 'PASS'
        look = n in (12, 18, 24) or (n > 24 and (n - 24) % 6 == 0)
        if not look: continue
        z = zstat(xs, R)
        if st == 'FORWARD_VALIDATING' and z <= -R['c_d']: st = 'FORWARD_FAILED'; continue
        if R['upgrade'] and st == 'FORWARD_VALIDATING' and n in (12, 24) and z >= R['b'] * math.sqrt(24 / n) and sig_ok(xs, R):
            st = 'UPGRADE_PENDING_24'
        if st == 'UPGRADE_PENDING_24' and n == 24: st = 'S1_EXCEPTION_EVIDENCE'
        elif st == 'FORWARD_VALIDATING' and n == 24: st = 'FORWARD_VALIDATING_NO_REJECTION_24'
        elif st == 'FORWARD_VALIDATING_NO_REJECTION_24' and n > 24 and z <= -R['c_d']: st = 'FORWARD_FAILED'
    return st, lit


def rules(PAR, ver):
    base = dict(sigma_design=PAR['sigma_design'], c=PAR['c'], c_d=PAR['c_d'], max_void_12=PAR.get('max_void_12', 2))
    v2 = dict(base, denom='v2', b=PAR['b_v2'], upgrade=True)
    if ver == 'A4': new = dict(base, denom='A', b=PAR['b_a'], upgrade=False); v2 = dict(v2, upgrade=False)
    elif ver == 'A1': new = dict(base, denom='A', b=PAR['b_a'], upgrade=True)
    elif ver == 'A2': new = dict(base, denom='A', b=PAR['b_a'], upgrade=True, sigcheck=True)
    elif ver == 'A3': new = dict(base, denom='H', b=PAR['b_a'], upgrade=True)
    else: raise ValueError(ver)
    return v2, new


def combined_state(items, PAR, ver):
    v2, new = rules(PAR, ver)
    s2, l2 = seq_rule(items, v2); sn, ln = seq_rule(items, new)
    assert l2 == ln
    return min((s2, sn), key=ORDER.index), s2, sn, l2
