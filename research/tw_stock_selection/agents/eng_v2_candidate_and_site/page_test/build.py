#!/usr/bin/env python3
"""本機頁面測試：把 index.html 包進假的 dash 執行環境（tpl.html；由 2026-11-16 的 harness 樣板加上 meta 與載入失敗模擬）。
用法：python3 -I -B build.py <index.html> <data 資料夾> <輸出 h.html> <d3.min.js 路徑> [情境 JSON]
情境 JSON：{"drop": ["thermo"], "error": ["rules"], "patch": {"thermo": "cold_full" | "bad"}}
真正的網站是 datasets/<id> → source.url 載入 JSON；本機沒有那個執行環境，這裡直接把 data/*.json 的 rows／meta 塞給假的 dash.data()。"""
import json, sys, os, re
page_path, data_dir, out, d3 = sys.argv[1:5]
sc = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
page = open(page_path, encoding='utf-8').read()
data, meta = {}, {}
for f in sorted(os.listdir(data_dir)):
    if f.endswith('.json'):
        j = json.load(open(os.path.join(data_dir, f), encoding='utf-8')); data[f[:-5]] = j['rows']; meta[f[:-5]] = j.get('meta')
for k in sc.get('drop', []):
    data.pop(k, None); meta.pop(k, None)
for k, how in sc.get('patch', {}).items():
    if how == 'bad':                      # 舊資料／壞資料：只剩月份，其他欄位缺、數字是字串或 null
        data[k] = [{'月份': '2026-10'}, {'月份': '2026-11', '飆股密度%': 'abc', '合成狀態': None}] if k == 'thermo' else [{'編號': 1}, {'編號': 2, '須知': None, '依據': ''}]
        meta[k] = None
    elif how == 'cold_full':              # 合成＝冷、完整月的畫面
        r = dict(data[k][-1]); r.update({'月份': '2026-12', '資料日': '2026-12-31', '完整月': '是', '飆股密度%': 0.5, '飆股密度_狀態': '冷', '大漲家數比例%': 1.2, '大漲家數比例_狀態': '冷',
                                         '右尾厚度%': 17.0, '右尾厚度_狀態': '溫', '合成狀態': '冷', '連續月數': 1}); data[k] = data[k] + [r]
    elif how == 'empty':
        data[k] = []
status = {k: 'error' for k in sc.get('error', [])}
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tpl.html'), encoding='utf-8').read()
scripts = re.findall(r'<script>(.*?)</script>', page, re.S)
html = re.sub(r'<script>.*?</script>', '', page, flags=re.S)
esc = lambda o: json.dumps(o, ensure_ascii=False).replace('</', '<\\/')
o = tpl.replace('/*DATA*/', 'window.__DATA=' + esc(data) + ';window.__META=' + esc(meta) + ';window.__STATUS=' + esc(status) + ';window.__SCRIPTS=' + esc(scripts) + ';').replace('<!--PAGE-->', html).replace('D3PATH', d3)
open(out, 'w', encoding='utf-8').write(o)
