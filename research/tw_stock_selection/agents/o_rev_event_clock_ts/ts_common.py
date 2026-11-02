#!/usr/bin/env python3
"""o_rev_event_clock_ts 共用：時間戳解析（git 快照 diff）、引擎載入（只讀 import o8 fa_common → V7 凍結面板的 V4 引擎）。
raw/ 下的 clone 一律當不可信資料：只用 git log / git show 讀日期與檔案內容，不執行其中任何程式。"""
import sys, io, json, gzip, hashlib, math, subprocess, datetime
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
RAW = HERE / 'raw'; DATA = HERE / 'data'; FWD = HERE / 'forward'
RES_F = HERE / 'results.json'
TPE = datetime.timezone(datetime.timedelta(hours=8))

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def canon_sha(o): return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [conv(v) for v in o]
    if isinstance(o, np.floating): return None if not np.isfinite(o) else float(o)
    if isinstance(o, float): return None if not math.isfinite(o) else o
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, (pd.Timestamp, pd.Period, datetime.datetime)): return str(o)
    return o

def jdump(o, f): Path(f).write_text(json.dumps(conv(o), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
def jload(f):
    f = Path(f); return json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
def save_key(k, o):
    r = jload(RES_F); r[k] = conv(o); jdump(r, RES_F)

def check_gate_prereg():
    p = HERE / 'gate_prereg.json'
    assert p.exists(), 'gate_prereg.json 不存在：先跑 prereg_gate.py'
    g = jload(p)
    assert g['script_sha256'] == sha(HERE / 'prereg_gate.py'), 'prereg_gate.py 在登記後被改過'
    assert canon_sha(g['rules']) == g['rules_sha256'], '登記規則被改過'
    return g

# ---------------- git ----------------
def git(repo, *a, binary=False):
    r = subprocess.run(['git', '-c', 'core.quotepath=off', '-C', str(repo), *a], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode('utf-8', 'replace')

def file_commits(repo, path):
    """回傳 [(hash, committer 時間 tz-aware)]，由舊到新；只列實際改到該路徑的 commit（含加入）。"""
    out = git(repo, 'log', '--format=%H %cI', '--', path)
    rows = [l.split() for l in out.splitlines() if l.strip()]
    rows = [(h, datetime.datetime.fromisoformat(t)) for h, t in rows]
    return rows[::-1]

def blob(repo, h, path):
    return git(repo, 'show', f'{h}:{path}', binary=True)

# ---------------- 解析 ----------------
def norm_code(s):
    s = str(s).strip().strip('"').strip()
    return s

def parse_tasi(b, ym):
    df = pd.read_csv(io.BytesIO(b), dtype=str)
    c0 = [c for c in df.columns if '代號' in c][0]; cv = [c for c in df.columns if '當月營收' in c][0]
    v = pd.to_numeric(df[cv].str.replace(',', ''), errors='coerce')
    codes = df[c0].map(norm_code)[v.notna()]
    return {(c, ym) for c in codes if c}

def roc_to_ym(x):
    x = str(x).strip()
    if not x.isdigit(): return None
    y, m = int(x[:-2]) + 1911, int(x[-2:])
    return f'{y:04d}-{m:02d}'

def parse_up_revenue_csv(b):
    df = pd.read_csv(io.BytesIO(b), dtype=str)
    ok = pd.to_numeric(df['rev_yoy'], errors='coerce').notna()
    return {(norm_code(c), roc_to_ym(r)) for c, r in zip(df.code[ok], df.rev_month[ok]) if roc_to_ym(r)}

def parse_up_long(b):
    raw = gzip.decompress(b) if b[:2] == b'\x1f\x8b' else b
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    ok = pd.to_numeric(df['revenue'], errors='coerce').notna()
    return {(norm_code(c), y) for c, y in zip(df.code[ok], df.ym[ok])}

def parse_rev_hist_ann(b):
    raw = gzip.decompress(b) if b[:2] == b'\x1f\x8b' else b
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    if 'ann_date' not in df: return {}
    d = df[df.ann_date.notna() & (df.ann_date != '')]
    return {(norm_code(c), y): a for c, y, a in zip(d.code, d.ym, d.ann_date)}

def first_seen(versions, gap_h=36.0):
    """versions：[(file_id, hash, T, set_of_pairs)] 依時間排序（同一來源所有檔案）。
    回傳 dict pair → dict(T, T_prev, file, valid, reason)；pair 以第一次出現為準。"""
    prev = {}; prevT = {}; out = {}
    by_commit = {}
    for fid, h, T, S in versions:
        new = S - prev.get(fid, set()) if fid in prev else set(S)
        new = {p for p in new if p not in out}
        by_commit.setdefault(h, []).append((fid, T, new, prevT.get(fid)))
        prev[fid] = S; prevT[fid] = T
        # 先暫存，批次規則需要同 commit 全部檔案
    # 依 commit 時間處理
    order = sorted(by_commit.items(), key=lambda kv: kv[1][0][1])
    out = {}
    for h, lst in order:
        cnt = {}
        for fid, T, new, Tp in lst:
            for c, y in new:
                if (c, y) in out: continue
                cnt.setdefault(c, set()).add(y)
        for fid, T, new, Tp in lst:
            for p in new:
                if p in out: continue
                c, y = p
                if Tp is None: rs = 'file_add_no_prev'
                elif (T - Tp).total_seconds() / 3600 > gap_h: rs = 'gap_gt_36h'
                elif len(cnt.get(c, ())) >= 2: rs = 'batch_backfill'
                else: rs = 'ok'
                out[p] = dict(T=T, T_prev=Tp, file=fid, commit=h, valid=rs == 'ok', reason=rs)
    return out

# ---------------- 交易日曆與公告日 ----------------
_ENG = {}
def engine():
    if _ENG: return _ENG
    sys.path.insert(0, str(AG / 'o_rev_freshness_attention'))
    import fa_common as FC   # noqa: E402（只讀 import；o8 V7 引擎載入）
    sys.path.pop(0)
    E = FC.engine()
    _ENG.update(E); _ENG['FC'] = FC
    return _ENG

def announce_day(T, tcal):
    """a(T)：台北日期；09:00 前取前一日；非交易日往後滾到下一個交易日。回傳 tcal 中的 Timestamp。"""
    t = T.astimezone(TPE)
    d = pd.Timestamp(t.date())
    if t.hour < 9: d -= pd.Timedelta(days=1)
    k = tcal.searchsorted(d)
    return tcal[k] if k < len(tcal) else None

def exec_day(T, tcal):
    """T 之後第一個交易日 09:00 開盤（台北）。"""
    t = T.astimezone(TPE)
    d = pd.Timestamp(t.date())
    if t.hour >= 9: d += pd.Timedelta(days=1)
    k = tcal.searchsorted(d)
    return tcal[k] if k < len(tcal) else None
