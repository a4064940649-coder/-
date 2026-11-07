import json, sys, os
H = os.path.dirname(os.path.abspath(sys.argv[0]))
a = json.load(open(os.path.join(H, '../../o_g8p2_protocol_refreeze/ro_end.json')))
b = json.load(open(os.path.join(H, '../ro_start.json')))
print('g8p2 end n', a.get('n_files'), a.get('listing_sha256', '')[:8], 'o16 start n', b['n_files'])
A, B = a['files'], b['files']
deleted = [k for k in A if k not in B]; added = [k for k in B if k not in A]
changed = [k for k in A if k in B and A[k]['sha'] != B[k]['sha']]
print('added', len(added), 'changed', len(changed), 'deleted', len(deleted), 'added+changed', len(added) + len(changed))
mt = sorted(B[k]['mtime'] for k in added + changed)
print('mtime range', mt[0], mt[-1])
