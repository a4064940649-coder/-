# 稽核執行器：python3 -I -B audit_run.py <role> <script.py> [args...]
# 以 sys.addaudithook 記錄每一次 open／import 到 logs/audit_<role>_<script>_<utc>.jsonl；
# role=A（重寫員）時，碰到禁讀清單直接拋錯中止（clean-room 守門）。
import sys, os, json, datetime, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
role, script = sys.argv[1], sys.argv[2]
sys.argv = [script] + sys.argv[3:]
utc = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
logp = os.path.join(HERE, 'logs', f'audit_{role}_{os.path.basename(script)[:-3]}_{utc}.jsonl')
FORBID_MOD = {'pclib','cz_engine','cz_v7x','v7env','x_exec','x_data','x_ideal','x_stats','ia_core','btlib',
              'v2_core','v4lib','o3_core','drj_core','st_core','erlib','bflib','o18lib','v7x_lib','ecore7','exlib','mlib','lhlib','avail','cz_lib','ia_lib','ilib'}
FORBID_PATH_SUB = ['plans_j7.csv', 'plans_ab7.csv', 'navs_oos_R', 'scratchpad/g6_sealed']
AGENTS = '/home/user/-/research/tw_stock_selection'
_buf = []
_seen = set()
def hook(ev, args):
    if ev == 'open':
        p = args[0]
        if isinstance(p, int): return
        p = os.fsdecode(p) if isinstance(p, (bytes, os.PathLike)) else str(p)
        ap = os.path.abspath(p)
        key = ('open', ap)
        if role == 'A':
            if any(s in ap for s in FORBID_PATH_SUB):
                raise PermissionError('clean-room 禁讀: ' + ap)
            if ap.endswith('.py') and ap.startswith(AGENTS) and not ap.startswith(HERE):
                raise PermissionError('clean-room 禁讀研究員 .py: ' + ap)
        if key not in _seen:
            _seen.add(key); _buf.append({'ev': 'open', 'path': ap, 'mode': str(args[1]) if len(args) > 1 else ''})
    elif ev == 'import':
        m = args[0]
        if role == 'A' and m.split('.')[0] in FORBID_MOD:
            raise ImportError('clean-room 禁 import: ' + m)
        key = ('import', m, str(args[1]))
        if key not in _seen:
            _seen.add(key); _buf.append({'ev': 'import', 'module': m, 'file': str(args[1])})
sys.addaudithook(hook)
sys.path.insert(0, HERE)
try:
    runpy.run_path(script, run_name='__main__')
finally:
    with open(logp, 'w') as f:
        for r in _buf: f.write(json.dumps(r, ensure_ascii=False) + '\n')
