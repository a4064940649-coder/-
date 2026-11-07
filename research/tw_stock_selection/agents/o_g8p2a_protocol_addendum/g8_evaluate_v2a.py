#!/usr/bin/env python3
"""G8P-2a 前瞻協定的機械判定器（G8_PROTOCOL_v2a.json 的一部分；凍結後只准修 bug，規則不得改；修 bug 須另立版本並重新外部錨定）。

唯一正式呼叫：python3 -I -B g8_evaluate_v2a.py status --official
（校準登錄沿用 G8P-2：python3 -I -B ../o_g8p2_protocol_refreeze/g8_evaluate_v2.py calibrate --official）
其他任何引數組合一律 NON_OFFICIAL（不得引用、不寫鏈）。測試旗標只在 NON_OFFICIAL 下生效。

判定＝較不利者（ORDER）of：
  C1 G8P-2 判定器原樣（import g8_evaluate_v2.py，sha 斷言；回溯／只往後兩種）
  C2 G8P-2 規則（分母 max(c·σ_d, s_n)、b、c_d）套在 v2a 序列上
  C3 新規則（協定 params.selected 版本；分母、b_a、c_d）套在 v2a 序列上
v2a 序列＝v2 的逐月決策線，但：adj 取全部適用 REAL_CALIB 的最大 Δ；open 月份逾期（(m+3) 月 26 日 00:00 台北前沒有已推上 origin 的鏈條目見證完成）→ V3 作廢；
期限內 open → 其後的檢視延後；整月缺列也算 open。
治理：status_chain.jsonl 雜湊鏈（只能追加；遠端截尾檢查）、吸收狀態與字面 G8 鎖定（RETRO_CHANGE 取較不利）、遠端錨定（GitHub push 事件時間）、
S1 證據須附 REAL_CALIB（否則 NO_REAL_CALIB、不得引用）、登錄簿 utc 逾 30 天或重複排定日 → INVALID。
只讀前瞻資料夾，不寫入 live_tracking 與 o_forward_infra_live。依賴：numpy、pandas、git、curl。"""
import sys, json, math, hashlib, datetime, subprocess, argparse, importlib.util, re, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
PROTO = HERE / 'G8_PROTOCOL_v2a.json'
MANIFEST = HERE / 'FREEZE_MANIFEST_v2a.json'
CHAIN = HERE / 'status_chain.jsonl'
RUNS = HERE / 'status_runs'
RAWR = HERE / 'raw' / 'remote_anchor'
V2_DIR = AG / 'o_g8p2_protocol_refreeze'
V2_PROTO, V2_MANIFEST, V2_EVAL, V2_REGISTRY = (V2_DIR / 'G8_PROTOCOL_v2.json', V2_DIR / 'FREEZE_MANIFEST_v2.json',
                                               V2_DIR / 'g8_evaluate_v2.py', V2_DIR / 'calib_registry.jsonl')
V2_SHA = dict(protocol='dcab4dbd255fb11e0ad8f4508397ae109cae9a47090eee17ab2ec7952981b993',
              manifest='23f45ce8ee8040a1fa8e721f76241d377f36e573e32a882d9e63c64aa4ed5c30',
              evaluator='1e4c0a16ab627e9418fef688bda4e27dfbb0540a53c6eacb954c7e6cfe22c38f')
LEDGER = AG / 'live_tracking/main_forward'
BACKLOG = AG / 'BACKLOG.md'
FORBID = [AG / 'live_tracking', AG / 'o_forward_infra_live']
REPO_SLUG = 'a4064940649-coder/-'
NOMINAL_FIRST_ANCHOR = '2026-11'
DEADLINE_UTC = '2026-11-10T05:30:00+00:00'
TZ_TPE = datetime.timezone(datetime.timedelta(hours=8))
JL = [f'J{j}' for j in range(13)]
ZERO = '0' * 64
ORDER = ['FORWARD_FAILED', 'FORWARD_VALIDATING_NO_REJECTION_24', 'FORWARD_VALIDATING', 'UPGRADE_PENDING_24', 'S1_EXCEPTION_EVIDENCE']
ABSORB = ('FORWARD_FAILED', 'S1_EXCEPTION_EVIDENCE')
SIG_MAX = 10.4


def sha_b(b): return hashlib.sha256(b).hexdigest()
def sha(p): return sha_b(Path(p).read_bytes())
def canon(o): return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
def utcnow(): return datetime.datetime.now(datetime.timezone.utc)
def entry_sha(e): return sha_b(canon({k: v for k, v in e.items() if k != 'entry_sha'}).encode())
def worst(states): return min((s for s in states if s), key=ORDER.index)


def load_v2():
    spec = importlib.util.spec_from_file_location('g8_evaluate_v2_ro', V2_EVAL)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- 遠端（origin＋GitHub push 事件）
class Remote:
    """git ls-remote／fetch／merge-base 判祖先；push 時間取 GitHub 伺服器端 activity（activity_type＝push）的 timestamp。
    events_file（測試）：JSON list of {timestamp, after, ref}，取代 API。"""
    def __init__(self, top, slug, save_dir=None, events_file=None, allow_fetch=True):
        self.top, self.slug, self.save_dir, self.events_file, self.allow_fetch = str(top), slug, save_dir, events_file, allow_fetch
        self._ev = None; self._pt = {}; self.log = []

    def git(self, *a, check=True, binary=False):
        r = subprocess.run(['git', '-C', self.top] + list(a), capture_output=True, timeout=120, text=not binary)
        if check and r.returncode != 0: raise RuntimeError(f'git {a[0]} rc={r.returncode} {r.stderr if not binary else ""}'.strip())
        return r.stdout

    def heads(self):
        out = self.git('ls-remote', 'origin')
        h = {}
        for l in out.strip().splitlines():
            s, ref = l.split('\t')
            if ref.startswith('refs/heads/'): h[ref] = s
        if not h: raise RuntimeError('ls-remote 沒有分支')
        return h

    def has(self, s):
        return subprocess.run(['git', '-C', self.top, 'cat-file', '-e', f'{s}^{{commit}}'], capture_output=True, timeout=30).returncode == 0

    def ensure(self, s, ref=None):
        if self.has(s): return True
        if not self.allow_fetch: return False
        subprocess.run(['git', '-C', self.top, 'fetch', '--no-tags', '-q', 'origin', ref or s], capture_output=True, timeout=300)
        return self.has(s)

    def is_anc(self, a, b):
        return subprocess.run(['git', '-C', self.top, 'merge-base', '--is-ancestor', a, b], capture_output=True, timeout=60).returncode == 0

    def first_commit(self, head, rel, needle=None):
        args = ['log', '--format=%H %cI']
        if needle: args.append(f'-S{needle}')
        lines = self.git(*(args + [head, '--', rel])).strip().splitlines()
        if not lines: return None, None
        h, t = lines[-1].split(' ', 1)
        return h, t

    def show(self, commit, rel):
        r = subprocess.run(['git', '-C', self.top, 'show', f'{commit}:{rel}'], capture_output=True, timeout=60)
        return r.stdout if r.returncode == 0 else None

    def events(self):
        if self._ev is not None: return self._ev
        if self.events_file:
            ev = json.loads(Path(self.events_file).read_text(encoding='utf-8'))
        else:
            ev = []; url = f'https://api.github.com/repos/{self.slug}/activity?per_page=100&activity_type=push'
            for page in range(40):
                r = subprocess.run(['curl', '-sS', '-m', '60', '-D', '-', '-H', 'Accept: application/vnd.github+json', url], capture_output=True, timeout=90)
                if r.returncode != 0: raise RuntimeError(f'curl rc={r.returncode}')
                raw = r.stdout
                hdr, _, body = raw.partition(b'\r\n\r\n')
                while body.startswith(b'HTTP/'): hdr, _, body = body.partition(b'\r\n\r\n')
                st = hdr.split(b'\r\n')[0].decode(errors='replace')
                if ' 200' not in st: raise RuntimeError(f'GitHub API {st}')
                fetched = utcnow().isoformat()
                if self.save_dir:
                    Path(self.save_dir).mkdir(parents=True, exist_ok=True)
                    fn = Path(self.save_dir) / f"activity_{utcnow().strftime('%Y%m%dT%H%M%SZ')}_p{page}.json"
                    fn.write_bytes(body)
                    self.log.append(dict(file=str(fn.relative_to(HERE)) if str(fn).startswith(str(HERE)) else str(fn), sha256=sha_b(body), fetched_utc=fetched, url=url))
                d = json.loads(body)
                ev += [dict(timestamp=e['timestamp'], after=e['after'], ref=e.get('ref'), id=e.get('id')) for e in d if e.get('activity_type') == 'push']
                m = re.search(rb'<([^>]+)>;\s*rel="next"', hdr)
                if not m or not d: break
                url = re.sub(r'/repositories/\d+/', f'/repos/{self.slug}/', m.group(1).decode())   # 代理不接受數字 ID 路徑，改寫成 owner/repo
        self._ev = sorted(ev, key=lambda e: e['timestamp'])
        return self._ev

    def push_time(self, commit):
        """含該 commit 的最早 push 事件時間（伺服器端）。"""
        if commit in self._pt: return self._pt[commit]
        best = None
        for e in self.events():
            if not self.ensure(e['after'], e.get('ref')): continue
            if self.is_anc(commit, e['after']):
                best = e; break
        self._pt[commit] = (pd.Timestamp(best['timestamp']).tz_convert('UTC').isoformat(), best['after'], best.get('id')) if best else None
        return self._pt[commit]


def remote_anchor(rm, man_p, backlog_p, chain_p, man_sha_hex):
    """回傳 dict：manifest 首次 commit／push 時間、BACKLOG 錨定行、遠端鏈內容。任何查詢失敗 → raise（REMOTE_UNVERIFIED）。"""
    top = Path(rm.top).resolve()
    heads = rm.heads()
    for ref, s in heads.items(): rm.ensure(s, ref)
    rel = lambda p: str(Path(p).resolve().relative_to(top))
    h0 = sorted(heads.items())[0]
    out = dict(heads=heads, problems=[], head_ref=h0[0], head=h0[1], chain_rel=rel(chain_p))   # manifest 不在遠端時，仍以第一個分支頭查 G8P-2 錨定
    # 找含 manifest 的 head
    mrel = rel(man_p)
    cand = []
    for ref, s in heads.items():
        c, ci = rm.first_commit(s, mrel)
        if c: cand.append((ref, s, c, ci))
    if not cand:
        out['problems'].append('ANCHOR_MANIFEST_NOT_ON_ORIGIN'); return out
    ref, head, c, ci = cand[0]
    blob = rm.show(c, mrel)
    out.update(head_ref=ref, head=head, manifest_first_commit=c, manifest_first_commit_local_cI=ci,
               manifest_content_matches=(blob is not None and sha_b(blob) == sha(man_p)),
               manifest_at_head_matches=(rm.show(head, mrel) is not None and sha_b(rm.show(head, mrel)) == sha(man_p)))
    if not out['manifest_content_matches'] or not out['manifest_at_head_matches']: out['problems'].append('ANCHOR_MANIFEST_CONTENT_MISMATCH')
    pt = rm.push_time(c)
    out['manifest_push'] = pt
    if pt is None: out['problems'].append('ANCHOR_MANIFEST_PUSH_UNKNOWN')
    # BACKLOG 錨定行（遠端內容）
    brel = rel(backlog_p)
    bh = rm.show(head, brel)
    out['backlog_remote_has_sha'] = bool(bh is not None and man_sha_hex in bh.decode('utf-8', 'replace'))
    if not out['backlog_remote_has_sha']: out['problems'].append('ANCHOR_BACKLOG_REMOTE_MISSING')
    else:
        bc, bci = rm.first_commit(head, brel, man_sha_hex)
        out['backlog_first_commit'] = bc; out['backlog_first_commit_local_cI'] = bci
        out['backlog_push'] = rm.push_time(bc) if bc else None
        if out['backlog_push'] is None: out['problems'].append('ANCHOR_BACKLOG_PUSH_UNKNOWN')
    # 遠端鏈
    crel = rel(chain_p)
    ch = rm.show(head, crel)
    out['remote_chain_lines'] = (ch.decode('utf-8').splitlines() if ch is not None else None)
    out['chain_rel'] = crel
    times = [x[0] for x in (out.get('manifest_push'), out.get('backlog_push')) if x]
    out['anchor_utc'] = max(times) if len(times) == 2 else None
    return out


def effective_first_anchor(anchor_utc):
    """錨定（origin push）早於期限 → 2026-11；否則＝push 後第一個完整錨定月（push 早於該月 10 日 13:30 台北者）。"""
    t = pd.Timestamp(anchor_utc).tz_convert(TZ_TPE)
    if pd.Timestamp(anchor_utc) < pd.Timestamp(DEADLINE_UTC): return NOMINAL_FIRST_ANCHOR
    m = pd.Period(t.strftime('%Y-%m'), 'M')
    cut = pd.Timestamp(m.year, m.month, 10, 13, 30, tz=TZ_TPE)
    return str(m if t < cut else m + 1)


# ---------------------------------------------------------------- 鏈
def read_chain(path, genesis_sha):
    probs = []
    if not Path(path).exists(): return [], ['CHAIN_MISSING']
    ents, prev, last_t = [], ZERO, None
    for i, l in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        if not l.strip(): probs.append(f'CHAIN_BLANK_LINE_{i}'); continue
        try: e = json.loads(l)
        except Exception: probs.append(f'CHAIN_BAD_JSON_{i}'); continue
        if e.get('seq') != len(ents): probs.append(f'CHAIN_SEQ_{i}')
        if e.get('prev_sha') != prev: probs.append(f'CHAIN_BROKEN_{i}')
        if entry_sha(e) != e.get('entry_sha'): probs.append(f'CHAIN_ENTRY_SHA_{i}')
        try:
            t = pd.Timestamp(e['utc'])
            if last_t is not None and t < last_t: probs.append(f'CHAIN_UTC_BACKWARDS_{i}')
            last_t = t
        except Exception: probs.append(f'CHAIN_UTC_{i}')
        prev = e.get('entry_sha'); ents.append(e)
    if not ents or ents[0].get('type') != 'GENESIS': probs.append('CHAIN_NO_GENESIS')
    elif ents[0].get('entry_sha') != genesis_sha: probs.append('CHAIN_GENESIS_MISMATCH')
    return ents, probs


def registry_extra_checks(ents):
    """(vi) 登錄簿：條目 utc 晚於 sched_date＋30 天、或同一排定日重複 → INVALID。"""
    probs, seen = [], {}
    for e in ents:
        sd = e.get('sched_date')
        if sd is None: continue
        if sd in seen: probs.append(f'REGISTRY_DUPLICATE_SCHED_{sd}')
        seen[sd] = 1
        try:
            if pd.Timestamp(e['utc']).tz_convert('UTC') > pd.Timestamp(sd).tz_localize('UTC') + pd.Timedelta(days=30): probs.append(f'REGISTRY_LATE_{sd}')
        except Exception: probs.append(f'REGISTRY_UTC_{sd}')
    return probs


# ---------------------------------------------------------------- 月份（v2a 序列）
def month_deadline(an):
    m = pd.Period(an, 'M') + 3
    return pd.Timestamp(m.year, m.month, 26, 0, 0, tz=TZ_TPE).tz_convert('UTC')


def month_rows_a(EV2, fwd, P2, first_anchor, calibs, mode, now, witness, active_from):
    """v2a 逐月：x_w＝min(excess_pp − adj_w, excess_stress_pp)，adj_w＝100·max(適用 Δ)·換手；有效 J ≥ 7 取中位數。
    月份範圍＝first_anchor 到 max(帳本最後錨定月, 期限已過的最後錨定月)；缺列或未全 complete＝open；
    錨定月 ≥ active_from 且 now > 期限且沒有見證 → V3。"""
    fr = pd.read_csv(Path(fwd) / 'forward_returns.csv')
    trp = Path(fwd) / 'trades.csv'
    tr = pd.read_csv(trp, dtype={'code': str}) if trp.exists() else pd.DataFrame(columns=['kind', 'line', 'date', 'shares', 'fill_px'])
    R = P2['rule']; VR = P2['void_rules']
    to = {j: EV2.turnover(fr, tr, j) for j in JL} if calibs else {}
    present = sorted(a for a in fr.anchor.astype(str).unique() if a >= first_anchor) if len(fr) else []
    last = pd.Period(present[-1], 'M') if present else None
    p = pd.Period(first_anchor, 'M')
    lastdue = None
    while month_deadline(str(p)) < now: lastdue = p; p += 1
    ends = [x for x in (last, lastdue) if x is not None]
    rows = []
    if not ends: return rows
    for per in pd.period_range(pd.Period(first_anchor, 'M'), max(ends), freq='M'):
        an = str(per)
        J = fr[(fr.anchor.astype(str) == an) & fr.line.isin(JL)] if len(fr) else fr
        complete = (len(J) == 13) and (J.status == 'complete').all()
        dl = month_deadline(an)
        if an >= active_from and now > dl and not witness(an, dl):
            rows.append(dict(anchor=an, state='void', void_kind='V3', deadline_utc=dl.isoformat(), ledger_complete=bool(complete))); continue
        if not complete:
            rows.append(dict(anchor=an, state='open', deadline_utc=dl.isoformat())); continue
        valid, voids = [], {}
        for _, r in J.iterrows():
            m, h = EV2.meta_of(fwd, r.line, an)
            v = EV2.void_reason(m, VR)
            if v: voids[r.line] = v
            else: valid.append(r)
        if len(valid) < R['jmed_min_valid_j']:
            rows.append(dict(anchor=an, state='void', void_kind='V0-V2', n_valid_j=len(valid), voids=voids)); continue
        xs = []
        for r in valid:
            ds = [float(c['delta_per_side']) for c in calibs
                  if mode == 'retro' or pd.Timestamp(r.start) > pd.Timestamp(c['valid_from']).tz_convert('UTC').tz_localize(None)]
            adj = 100.0 * max(ds) * to.get(r.line, {}).get(an, 0.0) if ds else 0.0
            xs.append(min(float(r.excess_pp) - adj, float(r.excess_stress_pp)))
        rows.append(dict(anchor=an, state='valid', n_valid_j=len(valid), voids=voids, x=float(np.median(xs)), j_p10=float(np.percentile(xs, 10))))
    return rows


def sequential_a(rows, R, active_from):
    """v2a 規則序貫（C2：denom v2；C3：新規則）。遇到 open 月份即停止（其後檢視延後）。active_from 之前的檢視不作用（照 G8P-2）。"""
    xs = []; st = 'FORWARD_VALIDATING'; ev = []; nvb = 0; ext = False; lit = None
    nv_all = sum(1 for r in rows if r['state'] == 'void')
    deferred = None
    for r in rows:
        if r['state'] == 'open': deferred = r['anchor']; break
        if r['state'] == 'void':
            if len(xs) < 12: nvb += 1
            continue
        xs.append(r['x']); n = len(xs)
        if st in ABSORB: continue
        act = r['anchor'] >= active_from
        if n == 12:
            if nvb > R['max_void_12']: ext = True; lit = 'UNDECIDABLE_EXTENDED'
            elif sum(xs) <= 0:
                lit = 'FAIL'
                if act: st = 'FORWARD_FAILED'; ev.append(dict(n=n, anchor=r['anchor'], event='LITERAL_G8_FAIL', S=sum(xs)))
                continue
            else: lit = 'PASS'
        if n == 24 and ext:
            if sum(xs) <= 0 or nv_all > 2 * R['max_void_12']:
                lit = 'FAIL'
                if act: st = 'FORWARD_FAILED'; ev.append(dict(n=n, anchor=r['anchor'], event='LITERAL_G8_FAIL_OR_UNDECIDABLE_EXT', S=sum(xs)))
                continue
            lit = 'PASS'
        look = n in (12, 18, 24) or (n > 24 and (n - 24) % 6 == 0)
        if not look or not act: continue
        z = zstat(xs, R)
        if st == 'FORWARD_VALIDATING' and z <= -R['c_d']:
            st = 'FORWARD_FAILED'; ev.append(dict(n=n, anchor=r['anchor'], event='DOWN_BOUNDARY', z=z)); continue
        if R['upgrade'] and st == 'FORWARD_VALIDATING' and n in (12, 24) and z >= R['b'] * math.sqrt(24 / n):
            sc = sigma_check(xs, R)
            if (not R.get('sigcheck')) or sc['ok']:
                st = 'UPGRADE_PENDING_24'; ev.append(dict(n=n, anchor=r['anchor'], event='UPGRADE_BOUNDARY', z=z, sigma_check=sc))
            else:
                ev.append(dict(n=n, anchor=r['anchor'], event='UPGRADE_BLOCKED_SIGMA', z=z, sigma_check=sc))
        if st == 'UPGRADE_PENDING_24' and n == 24:
            st = 'S1_EXCEPTION_EVIDENCE'; ev.append(dict(n=n, anchor=r['anchor'], event='S1_EVIDENCE_AT_24'))
        elif st == 'FORWARD_VALIDATING' and n == 24:
            st = 'FORWARD_VALIDATING_NO_REJECTION_24'; ev.append(dict(n=n, anchor=r['anchor'], event='NO_REJECTION_AT_24'))
        elif st == 'FORWARD_VALIDATING_NO_REJECTION_24' and n > 24 and z <= -R['c_d']:
            st = 'FORWARD_FAILED'; ev.append(dict(n=n, anchor=r['anchor'], event='DOWN_BOUNDARY', z=z))
    return dict(state=st, events=ev, literal=lit, n_valid=len(xs), S=float(sum(xs)), deferred_at=deferred)


def _sd(v):
    v = np.asarray(v, float); return float(np.std(v, ddof=1))


def _hac(v, L=6):
    y = np.asarray(v, float); n = len(y); y = y - y.mean()
    s = float((y * y).sum() / n)
    for k in range(1, min(L, n - 1) + 1):
        s += 2.0 * (1.0 - k / (L + 1.0)) * float((y[k:] * y[:-k]).sum()) / n
    return math.sqrt(max(s, 1e-12))


def zstat(xs, R):
    n = len(xs); S = float(np.sum(xs)); sn = _sd(xs)
    if R['denom'] == 'v2': sig = max(R['c'] * R['sigma_design'], sn)
    elif R['denom'] == 'A': sig = R['c'] * max(R['sigma_design'], sn)
    elif R['denom'] == 'H': sig = max(R['c'] * R['sigma_design'], _hac(xs, 6))
    else: raise ValueError(R['denom'])
    return S / (sig * math.sqrt(n))


def sigma_check(xs, R):
    x = np.asarray(xs, float)
    vals = [_sd(x)] + ([_sd(x[-12:])] if len(x) >= 12 else []) + ([_sd(x[-24:])] if len(x) >= 24 else [])
    chk = max(vals)
    grid, lab = R['sigma_grid'], R['sigma_label']
    cell = next((l for g, l in zip(grid, lab) if g >= chk - 1e-12), None)
    fu = R['false_up_max_by_sigma'].get(cell) if cell else None
    ok = bool(cell is not None and fu is not None and fu <= 0.05 and chk <= SIG_MAX + 1e-12)
    return dict(sigma_checked=chk, sigma_cell=cell or 'OUT_OF_GRID', cell_false_up_max=fu, ok=ok)


def rules_from(PA):
    pr = PA['params']
    base = dict(sigma_design=pr['sigma_design'], c=pr['c'], c_d=pr['c_d'], max_void_12=pr['g8_max_void_in_12'],
                sigma_grid=PA['oc']['sigma_grid'], sigma_label=PA['oc']['sigma_label'], false_up_max_by_sigma=PA['oc']['false_up_max_by_sigma'])
    sel = pr['selected']
    C2 = dict(base, denom='v2', b=pr['b_v2'], upgrade=(sel != 'A4'))
    if sel == 'A4': C3 = dict(base, denom='A', b=99.0, upgrade=False)
    else: C3 = dict(base, denom={'A1': 'A', 'A2': 'A', 'A3': 'H'}[sel], b=pr['b_a'], upgrade=True, sigcheck=(sel == 'A2'))
    return C2, C3


# ---------------------------------------------------------------- status
def status(a, official, argv):
    T = (lambda k: getattr(a, k) if not official else None)
    now = pd.Timestamp(T('test_now')) if T('test_now') else pd.Timestamp(utcnow())
    if now.tzinfo is None: now = now.tz_localize('UTC')
    now = now.tz_convert('UTC')
    proto_p = Path(T('test_protocol') or PROTO); man_p = Path(T('test_manifest') or MANIFEST)
    chain_p = Path(T('test_chain') or CHAIN); reg_p = Path(T('test_registry') or V2_REGISTRY)
    fwd = Path(T('test_forward') or LEDGER); backlog_p = Path(T('test_backlog') or BACKLOG)
    v2man_p = Path(T('test_v2_manifest') or V2_MANIFEST)
    problems, flags = [], []
    man = json.loads(man_p.read_text(encoding='utf-8')) if man_p.exists() else None
    if man is None: problems.append('MANIFEST_MISSING')
    if not proto_p.exists():
        out = dict(OFFICIAL=False, label='INVALID（不得引用）', problems=problems + ['PROTOCOL_MISSING'], state=None)
        if not a.quiet: print(json.dumps(out, ensure_ascii=False, indent=1))
        return out
    PA = json.loads(proto_p.read_text(encoding='utf-8'))
    if man and sha(proto_p) != man.get('protocol_sha256'): problems.append('PROTOCOL_SHA_MISMATCH')
    if man and sha(__file__) != man.get('evaluator_sha256'): problems.append('EVALUATOR_SHA_MISMATCH')
    for k, p in (('protocol', V2_PROTO), ('manifest', v2man_p if not official else V2_MANIFEST), ('evaluator', V2_EVAL)):
        if not p.exists() or sha(p) != V2_SHA[k]: problems.append(f'V2_{k.upper()}_SHA_MISMATCH')
    if 'V2_EVALUATOR_SHA_MISMATCH' in problems:
        out = dict(OFFICIAL=False, label='INVALID（不得引用）', problems=problems, state=None)
        if not a.quiet: print(json.dumps(out, ensure_ascii=False, indent=1))
        return out
    EV2 = load_v2()
    P2 = json.loads(V2_PROTO.read_text(encoding='utf-8'))
    # 登錄簿（v2 規則＋(vi)）
    v2man = json.loads(V2_MANIFEST.read_text(encoding='utf-8'))
    rents, rp = EV2.read_registry(reg_p, v2man); problems += rp
    problems += registry_extra_checks(rents)
    due = EV2.schedule(now.tz_localize(None) - pd.Timedelta(days=30))
    have = {e.get('sched_date') for e in rents}
    miss = [d for d in due if d not in have]
    if miss: problems.append(f'CALIB_MISSING {miss}')
    # 鏈
    cents, cp = read_chain(chain_p, man.get('status_chain_genesis_entry_sha') if man else None); problems += cp
    # 遠端錨定
    remote_ok, anch, anch_v2 = False, None, None
    if official or not a.test_skip_remote:
        top = Path(T('test_repo') or subprocess.run(['git', '-C', str(HERE), 'rev-parse', '--show-toplevel'], capture_output=True, text=True).stdout.strip())
        rm = Remote(top, REPO_SLUG, save_dir=(RAWR if official else T('test_raw_dir')), events_file=T('test_push_events'))
        try:
            anch = remote_anchor(rm, man_p, backlog_p, chain_p, sha(man_p))
            problems += anch['problems']
            # v2 錨定（同法：origin push 時間）
            v2rel = str(v2man_p.resolve().relative_to(Path(rm.top).resolve()))
            c2, ci2 = rm.first_commit(anch['head'], v2rel) if anch.get('head') else (None, None)
            anch_v2 = dict(first_commit=c2, local_cI=ci2, push=(rm.push_time(c2) if c2 else None))
            if not c2 or not anch_v2['push']: problems.append('V2_ANCHOR_REMOTE_MISSING')
            # 遠端鏈截尾檢查
            rl = anch.get('remote_chain_lines')
            local_lines = chain_p.read_text(encoding='utf-8').splitlines() if chain_p.exists() else []
            if rl is not None and local_lines[:len(rl)] != rl: problems.append('CHAIN_NOT_EXTENSION_OF_REMOTE')
            if rl is None: flags.append('CHAIN_NOT_ON_ORIGIN_YET')
            remote_ok = True
            anch['api_log'] = rm.log
        except Exception as ex:
            flags.append(f'REMOTE_UNVERIFIED {type(ex).__name__}: {ex}')
            anch = dict(error=str(ex))
    else:
        flags.append('REMOTE_SKIPPED_TEST')
    if remote_ok and anch.get('anchor_utc'):
        eff_a = effective_first_anchor(anch['anchor_utc'])
        if pd.Timestamp(anch['anchor_utc']) >= pd.Timestamp(DEADLINE_UTC): flags.append('ANCHOR_AFTER_DEADLINE')
    elif not remote_ok and not official and a.test_skip_remote:
        eff_a = T('test_first_anchor') or NOMINAL_FIRST_ANCHOR
    else:
        eff_a = None
    if remote_ok and anch_v2 and anch_v2.get('push'):
        eff_v2 = EV2.effective_first_anchor(anch_v2['push'][0])
    elif not remote_ok and not official and a.test_skip_remote:
        eff_v2 = T('test_v2_first_anchor') or NOMINAL_FIRST_ANCHOR
    else:
        eff_v2 = None
    # 見證（open 月份期限）
    wcache = {}
    def witness(an, dl):
        if an in wcache: return wcache[an]
        ok = False
        for e in cents:
            if e.get('type') != 'STATUS' or pd.Timestamp(e['utc']) > dl: continue
            mrec = (e.get('months') or {}).get(an)
            if not mrec or mrec[0] not in ('valid', 'void'): continue
            if mrec[0] == 'void' and mrec[2] == 'V3': continue
            if remote_ok:
                c, _ = rm.first_commit(anch['head'], anch['chain_rel'], e['entry_sha'])
                pt = rm.push_time(c) if c else None
                if pt is None or pd.Timestamp(pt[0]) > dl: continue
            elif official or not a.test_skip_remote:
                continue
            ok = True; break
        wcache[an] = ok
        return ok
    calibs = [e for e in rents if e.get('type') == 'REAL_CALIB' and e.get('applied')]
    C2, C3 = rules_from(PA)
    res = {}; comp = {}
    state = None; lit = None; rows_retro = []
    if eff_a is not None and eff_v2 is not None:
        first = min(eff_v2, eff_a)
        for mode in ('retro', 'forward'):
            r1, _ = EV2.month_rows(fwd, P2, eff_v2, [dict(c) for c in calibs], mode)
            s1, ev1, g81, xs1 = EV2.sequential(r1, P2)
            ra = month_rows_a(EV2, fwd, P2, first, calibs, mode, now, witness, eff_a)
            if mode == 'retro': rows_retro = ra
            s2 = sequential_a(ra, C2, eff_a); s3 = sequential_a(ra, C3, eff_a)
            res[mode] = dict(C1_v2_as_implemented=dict(state=s1, events=ev1, g8_literal=g81, n_valid=len(xs1)), C2_v2_rule_on_v2a_seq=s2, C3_new_rule=s3, rows=ra)
            comp[mode] = [s1, s2['state'], s3['state']]
        state = worst(comp['retro'] + comp['forward'])
        # 字面 G8：任一元件 FAIL → FAIL；PASS 只在 v2a 序列上的 C2、C3（兩種模式）都已判 PASS 時才發出（open 延後期間不發出 C1 的提前 PASS）
        fails, a_pass = False, True
        for mode in res:
            g = res[mode]['C1_v2_as_implemented']['g8_literal']
            if g and 'label' not in g and not g.get('passed', True): fails = True
            if g and 'extended' in g and not g['extended'].get('passed', True): fails = True
            for c in ('C2_v2_rule_on_v2a_seq', 'C3_new_rule'):
                if res[mode][c]['literal'] == 'FAIL': fails = True
                if res[mode][c]['literal'] != 'PASS': a_pass = False
        if state == 'FORWARD_FAILED' and any(e.get('event', '').startswith('LITERAL') for m in res for e in res[m]['C1_v2_as_implemented']['events']): fails = True
        lit = 'FAIL' if fails else ('PASS' if a_pass else None)
    # 鎖定與回溯
    retro = []
    if state is not None:
        locked = next((e for e in cents if e.get('type') == 'STATUS' and e.get('state') in ABSORB), None)
        if locked and locked['state'] != state:
            retro.append(f"STATE {locked['state']}→{state}"); state = worst([state, locked['state']])
        locked_lit = next((e for e in cents if e.get('type') == 'STATUS' and e.get('literal') in ('PASS', 'FAIL')), None)
        if locked_lit and locked_lit['literal'] != lit:
            retro.append(f"LITERAL {locked_lit['literal']}→{lit}"); lit = 'FAIL' if 'FAIL' in (lit, locked_lit['literal']) else locked_lit['literal']
        rec = {}
        for e in cents:
            if e.get('type') != 'STATUS': continue
            for an, v in (e.get('months') or {}).items():
                if an not in rec and v[0] in ('valid', 'void'): rec[an] = v
        cur = {r['anchor']: r for r in rows_retro}
        changed = []
        for an, v in rec.items():
            r = cur.get(an)
            now_v = (r['state'], r.get('x'), r.get('void_kind')) if r else ('missing', None, None)
            if now_v[0] != v[0] or (v[0] == 'valid' and (now_v[1] is None or abs(now_v[1] - v[1]) > 1e-9)): changed.append(an)
        if changed:
            retro.append(f'MONTHS {changed}')
            lr = []
            for r in rows_retro:
                v = rec.get(r['anchor'])
                if v is None: lr.append(r)
                elif v[0] == 'valid': lr.append(dict(anchor=r['anchor'], state='valid', x=v[1]))
                else: lr.append(dict(anchor=r['anchor'], state='void', void_kind=v[2]))
            for an in rec:
                if an not in cur:
                    v = rec[an]; lr.append(dict(anchor=an, state=v[0], x=v[1], void_kind=v[2]))
            lr = sorted(lr, key=lambda r: r['anchor'])
            sl2 = sequential_a(lr, C2, eff_a); sl3 = sequential_a(lr, C3, eff_a)
            res['locked_rows'] = dict(C2=sl2, C3=sl3)
            state = worst([state, sl2['state'], sl3['state']])
            for s_ in (sl2, sl3):
                if s_['literal'] == 'FAIL': lit = 'FAIL'
        if retro: flags.append('RETRO_CHANGE')
    if state == 'S1_EXCEPTION_EVIDENCE' and not calibs: flags.append('NO_REAL_CALIB')
    if state == 'S1_EXCEPTION_EVIDENCE' and any(r.get('void_kind') == 'V3' for r in rows_retro): flags.append('V3_PRESENT')
    invalid = bool(problems)
    remote_unverified = any(f.startswith('REMOTE_UNVERIFIED') for f in flags)
    is_off = bool(official and not invalid and not remote_unverified)
    lang = PA['judgement_language']
    if invalid: stext = lang['invalid']
    elif state is None: stext = None
    elif 'NO_REAL_CALIB' in flags or 'V3_PRESENT' in flags:
        stext = lang['not_citable'] + '（' + '、'.join(f for f in flags if f in ('NO_REAL_CALIB', 'V3_PRESENT')) + '）'
    else: stext = lang['states'].get(state)
    led = {f: (sha(fwd / f) if (fwd / f).exists() else None) for f in ('forward_returns.csv', 'trades.csv', 'fills_input.csv')}
    label = 'OFFICIAL' if is_off else ('INVALID（不得引用）' if (official and invalid) else 'NON_OFFICIAL（不得引用）')
    out = dict(OFFICIAL=is_off, official_call=bool(official), label=label, argv=argv, flags_cli={k: v for k, v in vars(a).items()},
               utc=utcnow().isoformat(), now_used=now.isoformat(), protocol_sha256=sha(proto_p), manifest_sha256=(sha(man_p) if man_p.exists() else None),
               evaluator_sha256=sha(__file__), v2_shas={k: sha(p) for k, p in (('protocol', V2_PROTO), ('manifest', V2_MANIFEST), ('evaluator', V2_EVAL))},
               registry_sha256=(sha(reg_p) if reg_p.exists() else None), registry_types=[e.get('type') for e in rents],
               calibs_applied=[e.get('entry_sha') for e in calibs], chain_n=len(cents), chain_last_sha=(cents[-1]['entry_sha'] if cents else None),
               ledger_sha256=led, remote_anchor=anch, v2_remote_anchor=anch_v2, effective_first_anchor_v2a=eff_a, effective_first_anchor_v2=eff_v2,
               problems=problems, flags=flags, retro_changes=retro, state=(None if invalid else state), state_if_valid=state, literal_g8=lit,
               state_text=stext, s1_real_calib_entry_shas=([e.get('entry_sha') for e in calibs] if state == 'S1_EXCEPTION_EVIDENCE' else None),
               components={m: v for m, v in comp.items()}, by_mode=res, selected_version=PA['params']['selected'],
               scope=PA['scope']['s1_evidence_scope'], historical_gate_note=lang['historical_gate'])
    s = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    write_chain = is_off or (not official and a.test_write_chain)
    if official:
        RUNS.mkdir(exist_ok=True)
        op = RUNS / f"status_{utcnow().strftime('%Y%m%dT%H%M%SZ')}_{'OFFICIAL' if is_off else ('INVALID' if invalid else 'NON_OFFICIAL')}.json"
    else:
        op = Path(a.out) if a.out else None
    if op is not None:
        op = op.resolve()
        assert not any(str(op).startswith(str(f.resolve())) for f in FORBID), '輸出不得寫入前瞻資料夾'
        op.write_text(s, encoding='utf-8')
    if write_chain and state is not None and not invalid:
        months = {r['anchor']: [r['state'], r.get('x'), r.get('void_kind')] for r in rows_retro}
        e = dict(seq=len(cents), type='STATUS', utc=utcnow().isoformat() if official else now.isoformat(), prev_sha=(cents[-1]['entry_sha'] if cents else ZERO),
                 protocol_sha256=sha(proto_p), evaluator_sha256=sha(__file__), ledger_sha256=led, months=months, state=state, literal=lit,
                 components=comp, flags=flags, effective_first_anchor_v2a=eff_a,
                 anchor_utc=(anch or {}).get('anchor_utc') if isinstance(anch, dict) else None,
                 output_sha256=(sha(op) if op is not None else sha_b(s.encode())), official_call=bool(official))
        e['entry_sha'] = entry_sha(e)
        assert not any(str(chain_p.resolve()).startswith(str(f.resolve())) for f in FORBID)
        with open(chain_p, 'a', encoding='utf-8') as fh: fh.write(canon(e) + '\n')
        out['chain_appended'] = e['entry_sha']
    if not a.quiet: print(s)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['status'])
    ap.add_argument('--official', action='store_true')
    for f in ('--test-forward', '--test-registry', '--test-protocol', '--test-manifest', '--test-chain', '--test-now', '--test-backlog',
              '--test-repo', '--test-push-events', '--test-raw-dir', '--test-first-anchor', '--test-v2-first-anchor', '--test-v2-manifest', '--out'):
        ap.add_argument(f)
    ap.add_argument('--test-skip-remote', action='store_true')
    ap.add_argument('--test-write-chain', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args(argv)
    official = argv == ['status', '--official']
    return status(a, official, argv)


if __name__ == '__main__':
    main()
