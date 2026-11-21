#!/usr/bin/env python3
"""開關全關時 cslib.Panel 逐位不變的檢查：改之前的 cslib（備份檔）與改之後的 cslib，各自載入 Panel() 與 Panel(exright_fix=True)，
比每個陣列的 sha256。用法：python3 -I -B verify_panel_bits.py <改之前的 cslib.py 備份> → verify/panel_bits.json"""
import sys, json, hashlib, subprocess, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
CACHE = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/offfix/cache')
if len(sys.argv) == 4:                                    # 子行程：<cslib 路徑> <0/1> child
    import numpy as np
    sp = importlib.util.spec_from_file_location('cslib', sys.argv[1]); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    m.TMP = CACHE
    P = m.Panel(exright_fix=bool(int(sys.argv[2])))
    out = {}
    for k, v in sorted(vars(P).items()):
        if isinstance(v, np.ndarray):
            out[k] = hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
    print(json.dumps(out)); sys.exit(0)
res = {}
for fix in (0, 1):
    h = [json.loads(subprocess.run([sys.executable, '-I', '-B', __file__, p, str(fix), 'child'], capture_output=True, text=True, check=True).stdout)
         for p in (sys.argv[1], str(AG / 'o_ext_canslim_xs' / 'cslib.py'))]
    res[f'exright_fix={fix}'] = dict(n_arrays=len(h[0]), same_keys=set(h[0]) == set(h[1]), diff=[k for k in h[0] if h[0][k] != h[1].get(k)])
res['pass'] = all(v['same_keys'] and not v['diff'] for v in res.values())
json.dump(res, open(HERE / 'verify' / 'panel_bits.json', 'w'), ensure_ascii=False, indent=1); print(res)
