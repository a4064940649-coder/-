#!/usr/bin/env python3
"""網站風險表口徑統一（2026-11-17）：用「正式引擎」重跑 S7／S9（S26 的來源）、S27、S42、XQ 純衛星（exright_engine_fix XAS）的原程式。
不改任何原研究檔、不改 cslib 預設值；做法同 s26_exright_robustness/runx.py：
  1. 在 import 各題 lib 之前設定開關：
       mode=exright  → cslib.EXRIGHT_FIX=True（DEADLINE_FIX、INS15_MASK 維持 False）＝舊數字的口徑，用來驗收「逐位重現」；
       mode=official → 三個開關全開（同 official_nav/run_official.py）。
  2. 快取、輸出全部改到 scratchpad/resync/（腳本 __file__ 指到 scratchpad/resync/run/<tag>/，原資料夾一個位元都不動）。
  3. 腳本裡的 assert 改成「記錄、不中止」（official 模式下，對照只修除權息公布值的驗收本來就會不符），記在 out/<tag>__soft.json。
  4. 墊片（shim）：共用引擎目前只有 cslib.rev_avail_days 讀 DEADLINE_FIX、只有 s10lib.make_signal_s10 讀 INS15_MASK。
     S7／S9 用 s4lib.make_signal（＝make_signal_s10 E0 的同一套進場，逐行同邏輯）、S9 與 S42 自己算 A_D 訊號日，開關會「部分失效」。
     為了讓正式口徑一致，在記憶體中包三個函式（開關關閉時一律直接呼叫原函式＝逐位相同）：
       a) s4lib.make_signal：INS15_MASK 開啟時改用本檔的逐行複製，只在 k＝k0＋shift 之後加一行 cslib.ins15_mask_row（位置同 make_signal_s10）。
       b) s9lib.signal_rows（conv='A'）：DEADLINE_FIX 開啟時，例外表月份訊號列取 max(A_D 列, 延後期限（含）後第一個交易日＋1)（同 S42 av_combine）。
          conv='B'（B11）不動（同 official_nav：B11 不受期限開關影響）。
       c) s42lib.av_family：同 b)。
用法：python3 -I -B harness.py <題目資料夾> <腳本> <exright|official> <tag> [--keep] [--link:名稱=目標] [-- 腳本參數...]"""
import sys, ast, shutil, json, re, time, resource
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parents[1]
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/resync')
OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)
T0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

argv = sys.argv[1:]
sargs = argv[argv.index('--') + 1:] if '--' in argv else []
argv = argv[:argv.index('--')] if '--' in argv else argv
study, script, mode, tag = argv[:4]
assert mode in ('exright', 'official')
keep = '--keep' in argv
CACHE = SCR / f'cache_{mode}'; CACHE.mkdir(parents=True, exist_ok=True)
SRC = AG / study
DST = SCR / 'run' / tag
if DST.exists() and not keep:
    shutil.rmtree(DST)
DST.mkdir(parents=True, exist_ok=True)
skip = {'navs.csv.gz', 'trades.csv.gz', 'results.json', 'results_posthoc.json', 'monthly_paths.csv.gz', 'build_meta.json', 'site_table.json'}
for f in SRC.iterdir():
    if f.is_file() and f.suffix != '.py' and f.name not in skip and not (keep and (DST / f.name).exists()):
        shutil.copy2(f, DST / f.name)
for a in argv:
    if a.startswith('--link:'):
        name, tgt = a[len('--link:'):].split('=', 1)
        lk = SCR / 'run' / name
        if lk.is_symlink() or lk.exists():
            lk.unlink()
        lk.symlink_to(tgt)


def addpath(folder):
    p = str(AG / folder)
    if p not in sys.path:
        sys.path.insert(0, p)


addpath('o_ext_canslim_xs')
import cslib                                                    # noqa: E402
assert cslib.EXRIGHT_FIX is False and cslib.DEADLINE_FIX is False and cslib.INS15_MASK is False, '共用引擎預設必須全關'
cslib.EXRIGHT_FIX = True
cslib.DEADLINE_FIX = cslib.INS15_MASK = (mode == 'official')
addpath(study)
src = (SRC / script).read_text()
src = re.sub(r"'/tmp/claude-0/[^']*/scratchpad/([A-Za-z0-9_]+)'", lambda m: repr(str(SCR / f'tmp_{mode}' / m.group(1))), src)
tree = ast.parse(src)
libs = sorted({a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names if re.fullmatch(r's\d+lib|stlib|cslib', a.name)} |
              {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and re.fullmatch(r's\d+lib|stlib|cslib', n.module)})
if script == 'run_engine.py':
    addpath('s19_xq_threshold'); libs.append('s19lib')
for m in libs:
    __import__(m)


def redirect():
    n = 0
    for name, mod in list(sys.modules.items()):
        if mod is None or not getattr(mod, '__file__', None) or str(AG) not in str(mod.__file__):
            continue
        for k, v in list(vars(mod).items()):
            if isinstance(v, Path) and 'scratchpad' in str(v) and 'scratchpad/resync' not in str(v):
                setattr(mod, k, CACHE); n += 1
    return n


nred = redirect()
assert cslib.EXRIGHT_FIX and cslib.DEADLINE_FIX == (mode == 'official') and cslib.INS15_MASK == (mode == 'official')


# ---------------------------------------------------------------- 墊片
def _dl_rows(P, months):
    """DEADLINE_FIX 開啟時：{例外月份: 延後期限（含）後第一個交易日＋1}（同 cslib.rev_avail_days 規則）；關閉時空 dict。"""
    if not cslib.DEADLINE_FIX:
        return {}
    ext = cslib.deadline_exceptions()
    return {m: int(np.searchsorted(P.IDX.values, np.datetime64(ext[m]), side='left')) + 1 for m in months if m in ext}


SHIM = dict(make_signal_ins15_calls=0, ins15_masked_rows=0, signal_rows_adjusted=0, av_family_adjusted=0)
mods = sys.modules
if 's4lib' in mods:
    s4lib = mods['s4lib']; _orig_ms = s4lib.make_signal

    def make_signal(P, univ, RT, depth=None, mom_lo=0.10, mom_hi=None, nsel=10, exit='top20', shift=0, keep_info=False):
        if not cslib.INS15_MASK:
            return _orig_ms(P, univ, RT, depth=depth, mom_lo=mom_lo, mom_hi=mom_hi, nsel=nsel, exit=exit, shift=shift, keep_info=keep_info)
        SHIM['make_signal_ins15_calls'] += 1
        S, av = RT['S'], RT['av']
        sig = np.zeros((P.T, P.N), bool); X = np.zeros((P.T, P.N), bool); rank = np.full((P.T, P.N), np.nan)
        info = []; pos_ma = {}; kmonth = {}
        for m in S.index:
            k0 = av[m]
            if k0 >= P.T or k0 < 61:
                continue
            s = S.loc[m].to_numpy()
            if int((univ[k0] & np.isfinite(s)).sum()) < 50:
                continue
            k = k0 + shift
            if k >= P.T:
                continue
            s2 = cslib.ins15_mask_row(P, m, k, s)              # ← 唯一加的一行（同 s10lib.make_signal_s10 的位置）
            if s2 is not s:
                SHIM['ins15_masked_rows'] += 1
            s = s2
            ok_ma = univ[k] & np.isfinite(s) & P.F_MA60[k]
            o_ma = np.flatnonzero(ok_ma)[np.argsort(-s[ok_ma], kind='stable')]
            r20 = P.RET20[k]
            pool = o_ma if depth is None else o_ma[:depth]
            okr = r20[pool] > mom_lo
            if mom_hi is not None:
                okr = okr & (r20[pool] <= mom_hi)
            tgt = pool[okr][:nsel]
            if exit == 'top20':
                keep_ = o_ma[:20]
            elif exit == 'f4':
                mid = o_ma[10:20]
                keep_ = np.concatenate([o_ma[:10], mid[r20[mid] > 0]])
            else:
                raise ValueError(exit)
            sig[k, tgt] = True; rank[k, tgt] = s[tgt]
            X[k, :] = True; X[k, keep_] = False
            kmonth[k] = m
            if keep_info:
                pos_ma[k] = {int(j): i + 1 for i, j in enumerate(o_ma)}
                nth = lambda n: float(s[o_ma[n - 1]]) if len(o_ma) >= n else float('nan')
                info.append(dict(month=str(m), k=int(k), sig_date=str(P.IDX[k].date()), n_ma=int(ok_ma.sum()), n_tgt=int(len(tgt)),
                                 targets=[str(P.COLS[j]) for j in tgt], surge_10th=nth(10), surge_15th=nth(15), surge_20th=nth(20)))
            else:
                info.append(dict(k=int(k), n_tgt=int(len(tgt))))
        return dict(sig=sig, rank=rank, X=X, SD=None, max_hold=None, info=info, pos_ma=pos_ma, kmonth=kmonth)
    s4lib.make_signal = make_signal

if 's9lib' in mods:
    s9lib = mods['s9lib']; _orig_sr = s9lib.signal_rows

    def signal_rows(P, months, conv, D=None, R=None):
        out, dl = _orig_sr(P, months, conv, D=D, R=R)
        if conv == 'A':
            for m, kd in _dl_rows(P, months).items():
                if kd > out[m]:
                    out[m] = kd; SHIM['signal_rows_adjusted'] += 1
        return out, dl
    s9lib.signal_rows = signal_rows

if 's42lib' in mods:
    s42lib = mods['s42lib']; _orig_af = s42lib.av_family

    def av_family(P, months, D):
        out = _orig_af(P, months, D)
        for m, kd in _dl_rows(P, months).items():
            if kd > out[m]:
                out[m] = kd; SHIM['av_family_adjusted'] += 1
        return out
    s42lib.av_family = av_family


# ---------------------------------------------------------------- assert 放行
class Soft(ast.NodeTransformer):
    def visit_Assert(self, node):
        msg = node.msg if node.msg is not None else ast.Constant(value=None)
        call = ast.Expr(value=ast.Call(func=ast.Name(id='__soft_assert__', ctx=ast.Load()),
                                       args=[ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=node.test),
                                             msg, ast.Constant(value=node.lineno)], keywords=[]))
        return ast.copy_location(call, node)


tree = ast.fix_missing_locations(Soft().visit(tree))
SOFT = []


def __soft_assert__(fn, msg, line):
    try:
        ok = bool(fn())
    except Exception as e:                                         # noqa: BLE001
        ok = False; msg = f'{msg} (例外 {e!r})'
    if not ok:
        s = repr(msg)
        SOFT.append(dict(line=line, msg=s[:3000]))
        print(f'[soft-assert 放行] 第 {line} 行：{s[:300]}', flush=True)


print(f'[{time.time()-T0:.0f}s] 執行 {study}/{script} mode={mode} → {DST}（libs {libs}，改快取 {nred} 處）', flush=True)
g = {'__name__': '__main__', '__file__': str(DST / script), '__soft_assert__': __soft_assert__, '__builtins__': __builtins__}
sys.argv = [str(DST / script)] + sargs
exec(compile(tree, str(SRC / script), 'exec'), g)
P = g.get('P')
info = dict(study=study, script=script, mode=mode, tag=tag, flags=dict(EXRIGHT_FIX=cslib.EXRIGHT_FIX, DEADLINE_FIX=cslib.DEADLINE_FIX, INS15_MASK=cslib.INS15_MASK),
            shim=SHIM, soft_asserts=SOFT, peak_mb=rss(), secs=round(time.time() - T0), panel_exright_fix=getattr(P, 'exright_fix', None))
json.dump(info, open(OUT / f'{tag}__soft.json', 'w'), ensure_ascii=False, indent=1, default=str)
print('完成', tag, '放行 assert', len(SOFT), 'shim', SHIM, 'peak', rss(), 'MB', flush=True)
