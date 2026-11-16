#!/usr/bin/env python3
"""比較兩個 results.json：數值最大絕對差、字串（判定）不同之處。用法：python3 -I -B jdiff.py a.json b.json [最多列幾筆]"""
import sys, json, math


def walk(a, b, p, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out['missing'].append(p + '/' + str(k)); continue
            walk(a[k], b[k], p + '/' + str(k), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out['len'].append(p); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f'{p}[{i}]', out)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        if a != b:
            out['str'].append((p, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
            d = abs(a - b)
            if d > out['maxd'][0]:
                out['maxd'] = (d, p, a, b)
            if d > 1e-9:
                out['nnum'] += 1
    else:
        if a != b:
            out['str'].append((p, a, b))


def diff(a, b):
    out = dict(missing=[], len=[], str=[], maxd=(0, '', None, None), nnum=0)
    walk(a, b, '', out)
    return out


if __name__ == '__main__':
    a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    o = diff(a, b)
    print('數值差 >1e-9 的個數', o['nnum'], '最大差', o['maxd'])
    print('缺鍵', len(o['missing']), o['missing'][:5], '長度不同', len(o['len']), o['len'][:5])
    print('字串／布林不同', len(o['str']))
    for x in o['str'][:n]:
        print('  ', x)
