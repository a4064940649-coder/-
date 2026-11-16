#!/usr/bin/env python3
"""S26 執行器：在「除權息修正開啟」（或關閉，用來核對）下，原封不動執行某一題的腳本（run.py／train.py／validate.py…），
只做三件事、都不改原檔：
  1. 在 import 該題 lib 之前設定 cslib.EXRIGHT_FIX；import 後把所有快取路徑改到 scratchpad/s26/cache。
  2. 該題資料夾的非程式檔（PREREG／FROZEN／results_train／特徵 csv…）複製到 scratchpad/s26/run/<tag>/，腳本的 __file__ 指到那裡，
     所以腳本寫出的 results.json／navs／trades 全部落在 scratchpad，原資料夾一個位元都不動。
  3. 腳本裡的 assert 改成「記錄、不中止」（修正開啟後，對照修正前公布值的驗收 A0 本來就會不符；其他驗收失敗會在 soft_asserts 列出）。
用法：python3 -I -B runx.py <題目資料夾> <腳本> <fix 0|1> [tag] [--no-frozen] [--keep]
  --no-frozen：不複製 FROZEN.json／FROZEN.sha256（重跑 train.py 看凍結選擇會不會變）。
輸出：scratchpad/s26/run/<tag>/ 下腳本的原始輸出；本資料夾 out/<tag>__soft.json（被放行的 assert）。"""
import sys, ast, shutil, json, re, io, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

study, script, fix = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
tag = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith('--') else f'{study.split("_")[0]}_{Path(script).stem}_fix{int(fix)}'
nofrozen = '--no-frozen' in sys.argv
keep = '--keep' in sys.argv          # 同一 tag 連續執行（例如 s24 build.py → analyze.py），不清空、不重複複製
SRC = common.AG / study
DST = common.SCR / 'run' / tag
if DST.exists() and not keep:
    shutil.rmtree(DST)
DST.mkdir(parents=True, exist_ok=True)
skip_out = {'navs.csv.gz', 'trades.csv.gz', 'results.json', 'results_posthoc.json'}
if Path(script).stem == 'train':
    skip_out |= {'results_train.json', 'cand_train.csv.gz'}
if Path(script).stem == 'validate':
    skip_out |= {'cand_valid.csv.gz'}
if nofrozen:
    skip_out |= {'FROZEN.json', 'FROZEN.sha256'}
for f in SRC.iterdir():
    if f.is_file() and f.suffix != '.py' and f.name not in skip_out and not (keep and (DST / f.name).exists()):
        shutil.copy2(f, DST / f.name)

cslib = common.set_fix(fix)
common.addpath(study)
src = (SRC / script).read_text()
# 腳本自己寫死的 scratchpad 快取路徑（例如 s24 build.py 的 S24）只在記憶體中改到 s26/cache
src = re.sub(r"scratchpad/[A-Za-z0-9_]+'", "scratchpad/s26/cache'", src)
tree = ast.parse(src)
libs = sorted({a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names if re.fullmatch(r's\d+lib|stlib|cslib', a.name)} |
              {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and re.fullmatch(r's\d+lib|stlib|cslib', n.module)})
for m in libs:
    __import__(m)
nred = common.redirect()
assert cslib.EXRIGHT_FIX == fix


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
        SOFT.append(dict(line=line, msg=s[:2000]))
        print(f'[soft-assert 放行] 第 {line} 行：{s[:300]}', flush=True)


g = {'__name__': '__main__', '__file__': str(DST / script), '__soft_assert__': __soft_assert__, '__builtins__': __builtins__}
common.log(f'執行 {study}/{script} fix={fix} → {DST}（libs {libs}，改快取 {nred} 處）')
sys.argv = [str(DST / script)]
exec(compile(tree, str(SRC / script), 'exec'), g)
P = g.get('P')
info = dict(study=study, script=script, fix=fix, tag=tag, soft_asserts=SOFT, peak_mb=common.rss(), secs=round(common.time.time() - common.T0),
            panel_exright_fix=getattr(P, 'exright_fix', None), exright_info=getattr(P, 'exright_info', None))
common.dump(f'{tag}__soft.json', info)
common.log('完成', tag, '放行 assert', len(SOFT))
