#!/usr/bin/env python3
"""產生交研究中心套用的 patch（不修改 independent_audit/ia_core.py）：ia_core_consideration.patch。
內容：ia_core.simulate 新增關鍵字參數 settle=None（None＝舊行為，freeze 預設不變）；settle 給定時改走 consideration 結算；
另加 build_settlement()。在 scratchpad 的副本上套用後驗證：(1) settle=None 時 C0／U_opt 與原 ia_core 逐日差 0；
(2) settle 給定時 U_opt 與 cz_engine consideration main 逐日差 0。python3 -I -B s13_patch.py"""
import sys, difflib, inspect, importlib.util, pickle, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cz_lib as Z
import cz_engine as CE
K = CE.K
src = (Z.AG / 'independent_audit' / 'ia_core.py').read_text(encoding='utf-8')
old_sig = 'def simulate(plan, buy, sell, t0, t1, freeze=True, delist_zero=False, ret_override=None):'
assert old_sig in src
new_sig = old_sig.replace('ret_override=None):', 'ret_override=None, settle=None):')
disp = ('    if settle is not None:   # 下市「收購／換股對價」口徑（o_engine_delist_consideration；settle＝build_settlement(events_v2)）\n'
        '        return _sim_consideration(plan, buy, sell, t0, t1, settle, ret_override)\n')
body_lines = src.split(old_sig)
doc_end = body_lines[1].index('"""', body_lines[1].index('"""') + 3) + 3
mod = body_lines[0] + new_sig + body_lines[1][:doc_end] + '\n' + disp + body_lines[1][doc_end:].lstrip('\n')
# 附加 consideration 實作（取自 cz_engine，改成模組內全域名）
add = inspect.getsource(CE._lt_engine) + '\n\n' + inspect.getsource(CE.build_settlement) + '\n\n' + inspect.getsource(CE._solve) + '\n\n' + inspect.getsource(CE._sim_cons)
add = add.replace('N = K.N; reb', 'NN = K.N; reb').replace('val = np.zeros(N); cash = 0.0 + 1.0', 'val = np.zeros(NN); cash = 0.0 + 1.0').replace('add_open = np.zeros(N)', 'add_open = np.zeros(NN)').replace('tgt = np.zeros(N); tgt[pk] = T\n            tov', 'tgt = np.zeros(NN); tgt[pk] = T\n            tov')
add = add.replace('E50 = K.E50.to_numpy()', 'E5n = K.E50.to_numpy()').replace('r50 = E50[t] / E50[t - 1] if np.isfinite(E50[t]) and np.isfinite(E50[t - 1])', 'r50 = E5n[t] / E5n[t - 1] if np.isfinite(E5n[t]) and np.isfinite(E5n[t - 1])')
add = add.replace('K.', '').replace('def _sim_cons(', 'def _sim_consideration(')
add = add.replace("TR = TRADED if TR is None else TR; IDX = IDX if IDX is None else IDX", "TR = TRADED if TR is None else TR; IDX = globals()['IDX'] if IDX is None else IDX")
add = add.replace("Cn = Cn if Cn is None else Cn; CI = CI if CI is None else CI", "Cn = globals()['Cn'] if Cn is None else Cn; CI = globals()['CI'] if CI is None else CI")
add = add.replace("On = On if On is None else On", "On = globals()['On'] if On is None else On")
mark = '# ---------------- 統計 ----------------'
assert mark in mod
mod = mod.replace(mark, '# ---------------- 下市對價口徑（o_engine_delist_consideration） ----------------\n' + add + '\n\n' + mark)
patch = ''.join(difflib.unified_diff(src.splitlines(True), mod.splitlines(True), 'a/research/tw_stock_selection/agents/independent_audit/ia_core.py',
                                     'b/research/tw_stock_selection/agents/independent_audit/ia_core.py'))
(HERE / 'ia_core_consideration.patch').write_text(patch, encoding='utf-8')
cp = Z.WORK / 'ia_core_patched.py'; cp.write_text(mod, encoding='utf-8')
spec = importlib.util.spec_from_file_location('ia_core_patched', cp); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
import pandas as pd
E = pd.read_csv(HERE / 'events_v2.csv', dtype={'code': str, 'successor': str})
PL = pickle.loads((Z.WORK / 'plans_ia.pkl').read_bytes())
bb, ss = K.rates(0.001); out = {}
for k in ('C0', 'U_opt'):
    v = PL[k]; plan = [(p[0], p[1]) for p in v['plan']]
    a, _ = K.simulate(plan, bb, ss, v['t0'], v['t1']); b, _ = M.simulate(plan, bb, ss, v['t0'], v['t1'])
    out[f'{k}|settle=None_vs_orig'] = float(np.max(np.abs(a.to_numpy() - b.to_numpy())))
    st1 = CE.build_settlement(E, 'main'); st2 = M.build_settlement(E, 'main')
    c, _ = CE.simulate(plan, bb, ss, v['t0'], v['t1'], mode='consideration', settle=st1); d, _ = M.simulate(plan, bb, ss, v['t0'], v['t1'], settle=st2)
    out[f'{k}|patched_cons_vs_cz_engine'] = float(np.max(np.abs(c.to_numpy() - d.to_numpy())))
assert max(out.values()) == 0.0, out
(HERE / 'patch_check.json').write_text(json.dumps(dict(checks=out, patch_sha256=Z.sha(HERE / 'ia_core_consideration.patch'),
                                                        base_ia_core_sha256=Z.sha(Z.AG / 'independent_audit' / 'ia_core.py')), ensure_ascii=False, indent=1), encoding='utf-8')
cp.unlink()
print(out)
