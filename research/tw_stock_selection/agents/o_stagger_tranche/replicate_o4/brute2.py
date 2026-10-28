import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
exec(open(pathlib.Path(__file__).parent / 'brute.py').read().split('# Note window')[0].replace("j = int(sys.argv[1])", "j = int(sys.argv[1])"))
cands = {}
for x in list(range(0, 40)) + [100 + j, 1000 + j, 10000 + j, 100003 * 0 + t, t, s.toordinal()]:
    cands[f'seed{x}'] = np.random.default_rng(x).random(C.N) < C.PCOL
for a in [(0, j), (0, j + 8), (0, t), (j, 0), (0, abs(j))]:
    try: cands[f'seq{a}'] = np.random.default_rng(list(a)).random(C.N) < C.PCOL
    except Exception as ex: pass
for i, (k, U) in enumerate(cands.items()):
    w, pk = window_ret(U, 50000 + i)
    if abs(w - target) < 1e-12: print('HIT', k)
