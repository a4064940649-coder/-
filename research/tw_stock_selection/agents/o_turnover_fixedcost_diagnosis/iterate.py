#!/usr/bin/env python3
"""迭代（只用挑選期 SEL：訊號日 ≤ 2019-04-30；100 萬一般；REAL_opt 為主並列 p90／carry）。
選擇準則在執行前寫死（不看 edge 結果選邊）：
  配對：不依任意排序者優先 → B（平均配對）為主；A2／A1／A0 並列（A0＝V7X 實際成交順序，規格字面）。
  價格分類：與實際來回成本的 Spearman 絕對值較大者（機制一致性）。
  持股槽貢獻：與 o19 定義一致者（原始加權貢獻 r/n）為主，超額版並列。
  大盤反轉：換股日可觀察者（0050 近 21 日報酬號 ≠ 前 21～126 日報酬號）為主，事後版並列。
  成本：ENG 為主，H05／H10 為假設區間。
→ iterations.json（含各版設定、挑選期量測與本檔 sha256）。
python3 -I -B iterate.py"""
import sys, json, pickle, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import diag_lib as DL; sys.path.pop(0)
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tfc')
T = pickle.load(open(SCR / 'T_normal_100萬.pkl', 'rb'))
VERS = [
 ('it1', dict(pairing='B', price_def='exit', cost='ENG', slot='excess', reversal='obs'), 'PREREG 草案（槽貢獻為超額版）'),
 ('it2', dict(pairing='A2', price_def='exit', cost='ENG', slot='excess', reversal='obs'), '邊際配對'),
 ('it3', dict(pairing='A1', price_def='exit', cost='ENG', slot='excess', reversal='obs'), '反向配對'),
 ('it4', dict(pairing='A0', price_def='exit', cost='ENG', slot='excess', reversal='obs'), 'V7X 實際成交順序配對（規格字面）'),
 ('it5', dict(pairing='B', price_def='min_A2', cost='ENG', slot='excess', reversal='obs'), '價格分類＝min(換出價, A2 夥伴價)'),
 ('it6', dict(pairing='B', price_def='exit', cost='ENG', slot='raw', reversal='obs'), '槽貢獻＝o19 原始加權貢獻'),
 ('it7', dict(pairing='B', price_def='exit', cost='ENG', slot='raw', reversal='post'), '反轉月份＝事後定義'),
 ('it8', dict(pairing='B', price_def='exit', cost='h10', slot='raw', reversal='obs'), '成本＝H10（一個跳動點加法模型）'),
]
out = {'scope': 'SEL（訊號日 2006-01～2019-04-30），seed 0，100 萬一般', 'criteria': __doc__, 'versions': []}
t0 = time.time()
X = DL.cut(T['exits'], 'SEL'); assert pd.to_datetime(X.signal).max() <= DL.SEL_END
for name, cfg, note in VERS:
    res = {v: DL.metrics(T, v, 'SEL', cfg, B=300, boot=(v == 'REAL_opt')) for v in ('REAL_opt', 'REAL_p90', 'REAL_carry')}
    o = res['REAL_opt']
    out['versions'].append(dict(name=name, cfg=cfg, note=note, sel=res))
    print(name, note, 'frac_lt', round(o['frac_lt'], 3), 'P3 ratio', round(o['P3']['ratio'], 2), 'P3 diff', round(o['P3']['diff_pp'], 1),
          'P4 share', round(o['P4']['share_1115_in_lt'], 3), 'P4 p', o['P4'].get('boot_p'), 'P6', o['P6']['holds'], round(time.time() - t0), flush=True)
# 價格分類準則
Xs = X[(X.variant == 'REAL_opt') & np.isfinite(X.edge_B)]
out['price_def_spearman'] = dict(exit=float(Xs.price.corr(Xs.rtc_B, method='spearman')),
                                 min_A2=float(pd.Series(np.fmin(Xs.price, Xs.partner_price_A2)).corr(Xs.rtc_A2, method='spearman')))
out['selected'] = dict(name='it6', cfg=dict(pairing='B', price_def='exit', cost='ENG', slot='raw', reversal='obs'),
                       reason='依事前準則：B 不依任意排序；exit 價格與來回成本 Spearman 絕對值較大；槽貢獻採 o19 定義；反轉採換股日可觀察版；成本 ENG。並列：A2／A0 配對、H05／H10、超額版槽貢獻、事後反轉。')
out['n_versions'] = len(VERS); out['utc'] = pd.Timestamp.now('UTC').isoformat(); out['iterate_py_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(HERE / 'iterations.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(out['price_def_spearman'], out['selected']['name'])
