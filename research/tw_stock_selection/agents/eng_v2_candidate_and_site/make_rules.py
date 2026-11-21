#!/usr/bin/env python3
"""產生網站「照做須知」資料檔 xq_scripts/site/data/rules.json（{rows, meta}）。文字與數字照 xq_scripts/README.md 第 32.1 節。
每一條的「須知」「依據」必須在 README 第 32.1 節的表格裡逐字找得到（找不到就報錯、不寫檔）；README 改了要重跑。
用法：python3 -I -B make_rules.py [輸出檔]（預設 xq_scripts/site/data/rules.json）→ 另寫 verify/rules_check.json"""
import sys, json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
README = REPO / 'xq_scripts' / 'README.md'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'xq_scripts' / 'site' / 'data' / 'rules.json'
TAG = '（未獨立重現）'
txt = README.read_text(encoding='utf-8')
a = txt.index('### 32.1'); b = txt.index('### 32.2')
sec = txt[a:b]
head = sec.split('\n', 1)[0]
assert '7 條操作須知' in head and 'R82' in head, head
rows = []
for ln in sec.split('\n'):
    m = re.fullmatch(r'\| (\d) \| (.+?) \| (.+?) \|', ln.strip())
    if not m:
        continue
    n, rule, basis = int(m.group(1)), m.group(2), m.group(3)
    unrep = basis.endswith(TAG)
    rows.append({'編號': n, '須知': rule, '依據': basis[:-len(TAG)] if unrep else basis,
                 '重現狀態': '未獨立重現' if unrep else '主要數字已獨立重現（R82）', '已獨立重現': '否' if unrep else '是'})
assert [r['編號'] for r in rows] == list(range(1, 8)), [r['編號'] for r in rows]
paras = [p.strip() for p in sec.split('\n') if p.strip() and not p.startswith('|') and not p.startswith('#')]
intro = paras[0]; bullets = [p[2:] for p in paras if p.startswith('- ')]
assert intro.startswith('2008～2026 連續帳戶 632 筆交易') and len(bullets) == 2 and bullets[1].startswith('出處'), (intro, bullets)
meta = [{'label': '產生程式', 'value': 'research/tw_stock_selection/agents/eng_v2_candidate_and_site/make_rules.py'},
        {'label': '標題', 'value': '這個策略靠少數大贏家：7 條操作須知'},
        {'label': '摘要', 'value': intro},
        {'label': '補充', 'value': bullets[0]},
        {'label': '重現說明', 'value': '第 1～5 條的主要數字已由獨立複製（R82）重現；第 6、7 條的數字未獨立重現。都是 2008～2026 歷史回測，不保證未來。'},
        {'label': '出處', 'value': 'xq_scripts/README.md 第 32.1 節（S82；R82）。' + bullets[1].replace('出處：', '').replace('`', '')},
        {'label': 'README 段落標題', 'value': head.lstrip('# ').strip()},
        {'label': 'make_rules.py', 'value': Path(__file__).read_text(encoding='utf-8'), 'type': 'code', 'lang': 'python'}]
chk = dict(n_rows=len(rows), all_rule_text_in_readme=all(r['須知'] in sec for r in rows), all_basis_text_in_readme=all(r['依據'] in sec for r in rows),
           unreplicated=[r['編號'] for r in rows if r['已獨立重現'] == '否'], intro_in_readme=intro in sec)
chk['pass'] = bool(chk['n_rows'] == 7 and chk['all_rule_text_in_readme'] and chk['all_basis_text_in_readme'] and chk['unreplicated'] == [6, 7] and chk['intro_in_readme'])
assert chk['pass'], chk
OUT.write_text(json.dumps({'rows': rows, 'meta': meta}, ensure_ascii=False, indent=1), encoding='utf-8')
(HERE / 'verify').mkdir(exist_ok=True)
(HERE / 'verify' / 'rules_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
print(chk); print(json.dumps(rows, ensure_ascii=False, indent=1))
