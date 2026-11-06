# 我的 OC 對研究員協定內 oc 表逐格比對
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib as L

f = sys.argv[1]
M = json.load(open(os.path.join(L.HERE, f)))
P = M['P']; C = M['cells']
oc = json.load(open(L.PROTO))['oc']
PB = 100000


def z(a, b):
    p = (a + b) / 2
    se = np.sqrt(max(p * (1 - p), 1e-9) * (1 / P + 1 / PB))
    return (a - b) / se


rows = []
def add(fam, key, mine, theirs):
    rows.append((fam, key, mine, theirs, z(mine, theirs)))


for dgp in ('D1', 'D2', 'D3', 'D3x', 'D4'):
    for sl in L.SLAB:
        for s in L.SGRID:
            mk = C[f"main|{dgp}|{sl}|a0.0|s{s}|m0"]
            sk = C[f"strict|{dgp}|{sl}|a0.0|s{s}|m1"]
            k = f"{dgp}|{sl}|{s}"
            if dgp in ('D1', 'D2', 'D3'):
                add('false_up_main', k, mk['V1_up'], oc['false_up_main'][k])
                add('up12_main', k, mk['V1_up12'], oc['up12_main'][k])
                add('false_up_strict', k, sk['V1_up'], oc['false_up_strict'][k])
                add('g8p1', k, sk['G8P1_up'], oc['g8p1_on_new_grid'][k])
            else:
                add('sens', k, mk['V1_up'], oc['sens'][k])
for dgp in ('D1', 'D2', 'D3'):
    for sl in L.SLAB:
        for a in (-1.0, -0.5, 0.0):
            for s, st in ((0.0, 's0'), (L.DW, 'sDW')):
                ck = (f"main|{dgp}|{sl}|a0.0|s0.0|m0" if (a == 0.0 and s == 0.0)
                      else f"dp|{dgp}|{sl}|a{a}|s{s}|m0")
                c = C[ck]; t = oc['down'][f"{dgp}|{sl}|a{a}|{st}"]
                add('down.dn', f"{dgp}|{sl}|a{a}|{st}", c['V1_dn'], t['dn'])
                add('down.fail', f"{dgp}|{sl}|a{a}|{st}", c['V1_fail'], t['fail'])
                add('down.up', f"{dgp}|{sl}|a{a}|{st}", c['V1_up'], t['up'])
        for a in (0.5, 1.0, 2.0):
            for s, st in ((0.0, 's0'), (L.DW, 'sDW')):
                c = C[f"dp|{dgp}|{sl}|a{a}|s{s}|m0"]
                add('power', f"{dgp}|{sl}|a{a}|{st}", c['V1_up'], oc['power'][f"{dgp}|{sl}|a{a}|{st}"])

zs = np.array([r[4] for r in rows])
print('cells', len(rows), 'max|z|', np.abs(zs).max(), 'n|z|>2', (np.abs(zs) > 2).sum(), 'n|z|>3', (np.abs(zs) > 3).sum())
fams = sorted(set(r[0] for r in rows))
summ = {}
for fm in fams:
    rr = [r for r in rows if r[0] == fm]
    zz = np.array([r[4] for r in rr])
    w = max(rr, key=lambda r: abs(r[4]))
    summ[fm] = dict(n=len(rr), maxabsz=float(np.abs(zz).max()), mean_z=float(zz.mean()), n_gt2=int((np.abs(zz) > 2).sum()), worst=w)
    print(f"{fm:16s} n={len(rr):3d} mean z={zz.mean():+.2f} max|z|={np.abs(zz).max():.2f} n>2={int((np.abs(zz)>2).sum())} worst={w[1]} mine={w[2]:.5f} theirs={w[3]:.5f}")

# 主要摘要
def mx(fam, pref=''):
    rr = [r for r in rows if r[0] == fam and r[1].startswith(pref)]
    a = max(rr, key=lambda r: r[2]); return a[1], a[2]
print('V1 main max', mx('false_up_main'), 'by dgp', [mx('false_up_main', d)[1] for d in ('D1', 'D2', 'D3')])
print('n main >0.05', sum(1 for r in rows if r[0] == 'false_up_main' and r[2] > 0.05))
print('by sigma', [round(max(r[2] for r in rows if r[0] == 'false_up_main' and r[1].split('|')[1] == sl), 4) for sl in L.SLAB])
s0 = [r[2] for r in rows if r[0] == 'false_up_main' and r[1].endswith('|0.0')]
print('s=0 range', min(s0), max(s0))
print('strict max', mx('false_up_strict'), 'n>0.05', sum(1 for r in rows if r[0] == 'false_up_strict' and r[2] > 0.05))
print('g8p1 max', mx('g8p1'), 'n>0.05', sum(1 for r in rows if r[0] == 'g8p1' and r[2] > 0.05))
print('g8p1 D1 s0', [round(C[f"strict|D1|{sl}|a0.0|s0.0|m1"]['G8P1_up'], 4) for sl in L.SLAB])
print('g8p1 D1 sd shift 0.3/0.577/0.78', C["strict|D1|6.14|a0.0|s0.3|m1"]['G8P1_up'],
      C["strict|D1|6.14|a0.0|s0.577|m1"]['G8P1_up'], C["strict|D1|6.14|a0.0|s0.78|m1"]['G8P1_up'])
print('g8p1 D3 s0', [round(C[f"strict|D3|{sl}|a0.0|s0.0|m1"]['G8P1_up'], 4) for sl in L.SLAB])
print('g8p1 D3x s0', [round(C[f"strict|D3x|{sl}|a0.0|s0.0|m1"]['G8P1_up'], 4) for sl in L.SLAB])
print('g8p1 D3x max', max(C[f"strict|D3x|{sl}|a0.0|s{s}|m1"]['G8P1_up'] for sl in L.SLAB for s in L.SGRID))
print('sens D3x max', mx('sens', 'D3x'), 'D4 max', mx('sens', 'D4'))
for d in ('D1', 'D2', 'D3'):
    c0 = C[f"main|{d}|6.14|a0.0|s0.0|m0"]; cm = C[f"dp|{d}|6.14|a-1.0|s0.0|m0"]
    print(d, 'dn_m1', cm['V1_dn'], 'dnraw', cm['V1_dnraw'], 'fdn0', c0['V1_dn'], c0['V1_dnraw'],
          'V3 dn_m1', cm['V3_dn'], 'V3 fdn0', c0['V3_dn'])
pw = {v: [C[f"dp|{d}|6.51|a1.0|s{L.DW}|m0"][v + '_up'] for d in ('D1', 'D2', 'D3')] for v in ('V1', 'V3', 'V4')}
print('power a1 s6.51 sDW', pw)
print('power sel min', {v: min(x) for v, x in pw.items()})
print('V1 a1 sDW max', max(C[f"dp|{d}|{sl}|a1.0|s{L.DW}|m0"]['V1_up'] for d in ('D1', 'D2', 'D3') for sl in L.SLAB))
print('V1 a2 sDW max', max(C[f"dp|{d}|{sl}|a2.0|s{L.DW}|m0"]['V1_up'] for d in ('D1', 'D2', 'D3') for sl in L.SLAB))
a1s0 = [C[f"dp|{d}|{sl}|a1.0|s0.0|m0"]['V1_up'] for d in ('D1', 'D2', 'D3') for sl in L.SLAB]
print('V1 a1 s0 range', min(a1s0), max(a1s0))
dsdw = [C[f"dp|{d}|6.14|a-1.0|s{L.DW}|m0"]['V1_dn'] for d in ('D1', 'D2', 'D3')]
print('K5 dn a-1 sDW sigma_d', dsdw)
print('K5 D1 dn a-1 s0 by sigma', [round(C[f"dp|D1|{sl}|a-1.0|s0.0|m0"]['V1_dn'], 3) for sl in L.SLAB])
print('K6 D3 fdn0 s0 by sigma', [round(C[f"main|D3|{sl}|a0.0|s0.0|m0"]['V1_dn'], 3) for sl in L.SLAB])
print('V3 main max', max(C[f"main|{d}|{sl}|a0.0|s{s}|m0"]['V3_up'] for d in ('D1','D2','D3') for sl in L.SLAB for s in L.SGRID))
print('V4 main max', max(C[f"main|{d}|{sl}|a0.0|s{s}|m0"]['V4_up'] for d in ('D1','D2','D3') for sl in L.SLAB for s in L.SGRID))
print('lit D1 a0 s=Dw sigma_d', C[f"dp|D1|6.14|a0.0|s{L.DW}|m0"]['lit'])
json.dump(dict(rows=rows, summ=summ), open(os.path.join(L.HERE, 'r3_compare_' + f), 'w'), ensure_ascii=False, indent=0)
