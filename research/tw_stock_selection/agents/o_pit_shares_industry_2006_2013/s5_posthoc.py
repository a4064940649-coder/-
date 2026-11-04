#!/usr/bin/env python3
"""事後（閘門已跑完、不改任何閘門數字）：
(1) 實作錯誤揭露與修正（只修一次）：s4 的「o12 回溯產業表 vs dated 快照」描述統計沒有統一名稱尾字（例：「建材營造」vs「建材營造業」），把命名差異算成產業不一致。
    修正方向：兩邊都去掉尾字「業」後再比（只影響描述，不影響 D2 閘門與金融旗標：金融旗標用的是已正規化的「金融保險業」）。
(2) 自我稽核：掃描本資料夾 .py 是否讀取 adj／ret 面板檔。
輸出寫入 results.json['posthoc']。"""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import plib as P
DR = pd.read_csv(P.SCR / 'd2_rows.csv', dtype={'code': str})
x = DR[DR.ind.notna() & DR.o12.notna()].copy()
n = lambda s: s.astype(str).str.replace('業$', '', regex=True).replace({'其他': '其他', '觀光餐旅': '觀光事業', '觀光事': '觀光事業'})
x['a'] = n(x.o12); x['b'] = n(x.ind)
mm = x[x.a != x.b]
out = dict(o12_vs_dated_fixed=dict(n_stock_months=len(mm), share=round(len(mm) / len(x), 4), n_stocks=int(mm.code.nunique()),
                                   pairs=mm.groupby(['code', 'o12', 'ind']).size().sort_values(ascending=False).head(30).reset_index().astype(str).to_dict('records')))
bad = {}
for f in sorted(p for p in P.HERE.glob('*.py') if p.name != 's5_posthoc.py'):   # 本檔含掃描用字串本身，排除
    s = f.read_text(encoding='utf-8')
    hits = re.findall(r"""['"](adj|ret)(\.parquet)?['"]|_rp\(['"](adj|ret)['"]\)|RET\b|\.Rn\b""", s)
    if hits: bad[f.name] = len(hits)
out['scan_adj_ret'] = dict(files=sorted(p.name for p in P.HERE.glob('*.py')), hits=bad, ok=not bad)
r = json.loads((P.HERE / 'results.json').read_text(encoding='utf-8')); r['posthoc'] = out
(P.HERE / 'results.json').write_text(P.dumps(r), encoding='utf-8')
print(out['o12_vs_dated_fixed']['n_stock_months'], out['o12_vs_dated_fixed']['share'], out['o12_vs_dated_fixed']['pairs'][:15], out['scan_adj_ret'])
