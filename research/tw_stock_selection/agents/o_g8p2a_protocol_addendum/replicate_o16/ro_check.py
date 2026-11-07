import json, os, hashlib, collections, sys
H = os.path.dirname(os.path.abspath(sys.argv[0]))
end = json.load(open(os.path.join(H, '..', 'ro_end.json')))
st = json.load(open(os.path.join(H, '..', 'ro_start.json')))
tops = collections.Counter(k.split('/')[0] + '/' + k.split('/')[1] for k in end['files'])
print(tops)
for base in ['/home/user/-/research/tw_stock_selection', '/home/user/-/research/tw_stock_selection/agents', '/home/user/-']:
    k0 = next(iter(end['files']))
    if os.path.exists(os.path.join(base, k0)):
        break
print('base', base)
# recompute hashes for listed files now
same = diff = miss = 0
for k, v in end['files'].items():
    p = os.path.join(base, k)
    if not os.path.exists(p): miss += 1; continue
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    if h == v['sha']: same += 1
    else: diff += 1
print('now vs ro_end: same', same, 'diff', diff, 'missing', miss)
print('start==end files dict identical:', st['files'] == end['files'])
# listing sha guess: sha256 of canonical json of files dict?
for cand in [json.dumps(end['files'], sort_keys=True), json.dumps(end['files'], sort_keys=True, separators=(',', ':')),
             '\n'.join(f"{k} {v['sha']}" for k, v in sorted(end['files'].items())),
             '\n'.join(f"{k}\t{v['sha']}" for k, v in sorted(end['files'].items())) ,
             json.dumps({k: v['sha'] for k, v in sorted(end['files'].items())}, sort_keys=True, separators=(',', ':'))]:
    print(hashlib.sha256(cand.encode()).hexdigest()[:8])
fr = [os.path.join(base, k) for k in end['files'] if k.endswith(('forward_returns.csv', 'trades.csv'))]
for p in fr:
    n = sum(1 for _ in open(p)) - 1
    print(p.replace(base + '/', ''), 'rows', n)
