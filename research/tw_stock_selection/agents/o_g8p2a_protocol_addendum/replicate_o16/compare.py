import json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import rlib as R
H = os.path.dirname(os.path.abspath(sys.argv[0]))
raw = json.load(open(os.path.join(H, 'rep_raw.json')))
O = R.P_['oc']
res = {}
q = raw['setA_q']
ba = max(R.BV2, max(q.values()))
top = sorted(q.items(), key=lambda kv: -kv[1])[:4]
res['b_a'] = dict(mine=ba, theirs=R.BA, diff=ba - R.BA, top=[[k, round(v, 4)] for k, v in top])
res['E3'] = {k: round(v, 5) for k, v in raw['E3'].items()}
B = {}
for kind, key, r in raw['setB']:
    B.setdefault(kind, {})[key] = r
def mk(k):
    d, sl, s, a, m = k.split('|'); return f"{d}|{sl}|{s}"
main = B['main']
fu = {mk(k): v['up_comb'] for k, v in main.items()}
mx = max(fu, key=fu.get)
mism = [(c, fu[c], O['false_up_main'][c]) for c in fu if abs(fu[c] - O['false_up_main'][c]) > 1e-9]
res['A1_main'] = dict(n=len(fu), max=fu[mx], cell=mx, n_gt_05=sum(v > .05 for v in fu.values()),
                      n_cells_mismatch_vs_protocol=len(mism), max_abs_diff=max([abs(a - b) for _, a, b in mism], default=0.0),
                      viol_total=sum(int(v['viol_up'] + v['viol_fail']) for kd in B.values() for v in kd.values()),
                      max_by_dgp={d: max(v for c, v in fu.items() if c.startswith(d + '|')) for d in R.MAIN},
                      s0_range=[min(v for c, v in fu.items() if c.endswith('|0.0')), max(v for c, v in fu.items() if c.endswith('|0.0'))])
v2 = {mk(k): v['up_v2'] for k, v in main.items()}
mx2 = max(v2, key=v2.get)
prim = [c for c in v2 if c.split('|')[0] in ('D1', 'D2', 'D3')]
res['v2_same'] = dict(max=v2[mx2], cell=mx2, n_gt_05=sum(v > .05 for v in v2.values()), primary135_max=max(v2[c] for c in prim),
                      n_mismatch=sum(abs(v2[c] - O['v2_same_paths']['false_up_main'][c]) > 1e-9 for c in v2))
b0 = {mk(k): v['up_A1b0'] for k, v in main.items()}
res['A1b0'] = dict(primary135_max=max(b0[c] for c in prim), max=max(b0.values()), n_gt_05=sum(v > .05 for v in b0.values()),
                   AR10_s078={sl: b0[f"AR10|{sl}|0.78"] for sl in ['8.4', '9.0', '9.5', '10.4']})
st = {mk(k): v['up_comb'] for k, v in B['strict'].items()}
ms = max(st, key=st.get)
res['strict'] = dict(max=st[ms], cell=ms, n_gt_05=sum(v > .05 for v in st.values()))
se = {mk(k): v['up_comb'] for k, v in B['sens'].items()}
res['sens'] = {d: max(v for c, v in se.items() if c.startswith(d + '|')) for d in R.SENS}
# power
pw = B['power']
def g(kind, d, sl, s, a, f):
    return B[kind][f"{d}|{sl}|{s}|a{a}|m0"][f]
res['power'] = dict(
    D1_651_DW_a1=dict(A1=g('power', 'D1', '6.51', 'DW', 1.0, 'up_comb'), v2=g('power', 'D1', '6.51', 'DW', 1.0, 'up_v2')),
    a1_s651_sDW_by_dgp={d: g('power', d, '6.51', 'DW', 1.0, 'up_comb') for d in R.MAIN})
for a in (0.5, 1.0, 2.0):
    for s in ('DW', '0.0'):
        vals = [v['up_comb'] for k, v in pw.items() if f"|{s}|a{a}|" in k]
        vv2 = [v['up_v2'] for k, v in pw.items() if f"|{s}|a{a}|" in k]
        res['power'][f"a{a}|{s}"] = [min(vals), max(vals)]
        res['power'][f"v2_a{a}|{s}"] = [min(vv2), max(vv2)]
res['power']['n_cells_A1_gt_v2'] = sum(v['up_comb'] > v['up_v2'] for v in pw.values())
pm = min(res['power']['a1_s651_sDW_by_dgp'], key=res['power']['a1_s651_sDW_by_dgp'].get)
res['power']['min'] = [pm, res['power']['a1_s651_sDW_by_dgp'][pm]]
# down
dn = B['down']
for f in ('dn_comb', 'dn_comb_nolit', 'dn_v2', 'dn_v2_nolit'):
    res['down_' + f] = dict(
        m1_s0_sigd={d: round(g('down', d, '6.14', '0.0', -1.0, f), 5) for d in R.MAIN},
        fdn0_s0_sigd={d: round(g('down', d, '6.14', '0.0', 0.0, f), 5) for d in R.MAIN},
        m1_sDW_range=[min(v[f] for k, v in dn.items() if '|DW|a-1.0' in k), max(v[f] for k, v in dn.items() if '|DW|a-1.0' in k)],
        fdn0_max=max(((v[f], k) for k, v in dn.items() if '|a0.0|' in k)),
        n_fdn0_gt_030=sum(v[f] > .30 for k, v in dn.items() if '|a0.0|' in k),
        n_fdn0_gt_030_D5=sum(v[f] > .30 for k, v in dn.items() if '|a0.0|' in k and k.startswith('D5|')))
json.dump(res, open(os.path.join(H, 'rep_results.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False, indent=1))
