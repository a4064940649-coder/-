# AST 靜態掃描：列出本資料夾每個 .py 的 import 與字串常數中的路徑，檢查禁用模組
import ast, os, json, sys, hashlib, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
FORBID = {'pclib','cz_engine','cz_v7x','v7env','x_exec','x_data','x_ideal','x_stats','ia_core','btlib','v2_core','v4lib','o3_core','drj_core','st_core','erlib','bflib','o18lib','v7x_lib','ecore7','exlib','mlib','ilib'}
out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': {}}
bad = []
for f in sorted(os.listdir(HERE)):
    if not f.endswith('.py'): continue
    src = open(os.path.join(HERE, f)).read()
    tree = ast.parse(src)
    imps, strs = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names: imps.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            imps.add(n.module or '')
        elif isinstance(n, ast.Call) and getattr(n.func, 'id', '') == '__import__':
            bad.append((f, '__import__ dynamic'))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and ('/' in n.value and ('.py' in n.value or 'agents' in n.value)):
            strs.add(n.value[:200])
    hit = sorted(m for m in imps if m.split('.')[0] in FORBID)
    if hit: bad.append((f, hit))
    out['files'][f] = {'sha256': hashlib.sha256(src.encode()).hexdigest(), 'imports': sorted(imps), 'path_strings': sorted(strs), 'forbidden_hits': hit}
out['violations'] = bad
json.dump(out, open(os.path.join(HERE, 'logs', sys.argv[1] if len(sys.argv) > 1 else 'ast_scan.json'), 'w'), ensure_ascii=False, indent=1)
print('violations', bad)
