#!/usr/bin/env python3
"""兩份名單（make_list.py 的 json 輸出）逐欄比對。用法：compare_lists.py <a.json> <b.json> [--json out]
比：狀態、訊號日、執行日、股池家數、L_MA 前 20 名每一欄、F1 每一欄、換股計畫（plan）整包。meta 只列出不比。結束碼 0＝全部相同，1＝有不同。"""
import json, sys


def flat(o, pre=''):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f'{pre}.{k}' if pre else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f'{pre}[{i}]')
    else:
        yield pre, o


def compare(a, b):
    out = dict(head={}, top20=[], f1=[], plan=[])
    for k in ('status', 'signal_month', 'rev_month', 'sig_date', 'exec_date', 'n_univ', 'n_univ_surge', 'n_ma', 'e50_close'):
        if a.get(k) != b.get(k):
            out['head'][k] = [a.get(k), b.get(k)]
    for sec in ('top20', 'f1', 'plan'):
        fa, fb = dict(flat(a.get(sec))), dict(flat(b.get(sec)))
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k, '<無>') != fb.get(k, '<無>'):
                out[sec].append([k, fa.get(k, '<無>'), fb.get(k, '<無>')])
    out['top20_codes_same'] = [r['code'] for r in a.get('top20', [])] == [r['code'] for r in b.get('top20', [])]
    out['f1_codes_same'] = [r['code'] for r in a.get('f1', [])] == [r['code'] for r in b.get('f1', [])]
    out['identical'] = not (out['head'] or out['top20'] or out['f1'] or out['plan'])
    return out


def main():
    a, b = json.load(open(sys.argv[1], encoding='utf-8')), json.load(open(sys.argv[2], encoding='utf-8'))
    r = compare(a, b)
    print(f"前 20 名代號順序相同：{r['top20_codes_same']}；F1 代號相同：{r['f1_codes_same']}；全部欄位相同：{r['identical']}")
    for sec in ('head', 'top20', 'f1', 'plan'):
        d = r[sec]
        if d:
            print(f'[{sec}] 不同 {len(d)} 處：', json.dumps(d if isinstance(d, dict) else d[:15], ensure_ascii=False))
    if '--json' in sys.argv:
        json.dump(r, open(sys.argv[sys.argv.index('--json') + 1], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    sys.exit(0 if r['identical'] else 1)


if __name__ == '__main__':
    main()
