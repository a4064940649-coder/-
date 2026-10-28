#!/usr/bin/env python3
"""步驟 4a：把 iterations.json 的選定模型與其他元件寫成 data/model_calib.json（容量模擬器唯一讀的模型檔）。只用 CAL 資料與探索結果；不看策略報酬。
元件：
  impact：選定模型（M3V：成本＝0.1% 基礎滑價＋b·σ·(Q/V60)^δ，截距 a 只用於擬合，不進成本——事前規則：所有模型都保留 o5 的 0.1% 基礎滑價，
          模型只提供隨規模變動的衝擊項）；p5／p10／p90／p95 取自聯合 bootstrap 該分位那次重抽的 (δ, b)。
  oa：開盤集合競價量的事前預測（iterations round 3 的 OA 模型，值口徑 OA_hat＝smear·V60·exp(β·x)），只用在 20% 參與上限。
  drift、odd：沿用 o5 model_spec.json（9:10 價格對開盤、零股買賣成本分桶；零股賣成本下限 0 在模擬器處理）。
  zcap：零股量上限（模型 Z）：2020-10-26 起，每檔每日零股成交上限＝part × 預測日零股成交值（＝十分位中位數占比 × V60）；
        part＝1.0（主欄），0.2（敏感度）。超過的部分順延到下一個交易日。9:10 首次撮合本身沒有資料（無法估計）。
  e2：E2（新進股一半 09:30 前成交）的價格＝開盤 × exp(−prem_hat(跳空分桶))，prem_hat＝CAL 各跳空桶的開盤溢價平均。
  sens：非判定欄的結構敏感度模型：M2（Y、無截距平方根、V60）、M1（κ_open、OA_hat）、VAL 重估的 M3V、o5 原式（0.5σ√(Q/OA_o5)）。
用法：python3 -I -B freeze_model.py
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import calib_lib as C
import iterate as IT_

IT = json.load(open(HERE / 'iterations.json'))
EXP = json.load(open(HERE / 'explore.json'))
sel = IT['selected']; assert sel['version'] == 'v4' and sel['params']['model'] == 'M3V'
bs = sel['bootstrap_CAL']['replicate_at_quantile']
r3 = [r for r in IT['rounds'] if r['round'] == 3][0]
o5 = json.load(open(C.DATA / 'model_spec.json'))
allv = {v['version']: v for r in IT['rounds'] for v in r['versions']}

X, R = IT_.load()
X['OA_hat'] = IT_.oa_hat(X, r3['oa_model'])
cal = X[X.U & (X.period == 'CAL')]
edges = [-0.03, -0.01, -0.0025, 0.0025, 0.01, 0.03]
k = np.searchsorted(edges, cal.gap.to_numpy(), side='right')
prem = [float(cal.prem[k == i].mean()) for i in range(len(edges) + 1)]
odd_share = {int(kk): float(v) for kk, v in pd.Series(EXP['oddlot_2026']['daily_odd_value_by_dec']).items()}
o = C.oddlot(); p = o[o.vrank_l1 <= 300].copy(); p['dec'] = np.ceil(p.vrank_l1 / 30)
share_dec = {int(kk): float(v) for kk, v in (p.amount / p.V60_l1).groupby(p.dec).median().items()}

M = dict(
    version='o6_calib_v4', selected_version=sel['version'],
    impact=dict(model='M3V', formula='imp = base + b·σ·(Q/V60)^δ；Q＝張數×1000×前收；V60、σ＝t−1', base=0.001,
                point=dict(delta=sel['params']['delta'], b=sel['params']['b']), a_fit_only=sel['params']['a'],
                p5=dict(delta=bs['p5']['delta'], b=bs['p5']['b']), p10=dict(delta=bs['p10']['delta'], b=bs['p10']['b']),
                p90=dict(delta=bs['p90']['delta'], b=bs['p90']['b']), p95=dict(delta=bs['p95']['delta'], b=bs['p95']['b'])),
    oa=dict(vars=r3['oa_model']['vars'], coef=r3['oa_model']['coef'], smear=r3['oa_model']['smear'], cap=0.20,
            note='OA_hat＝smear·V60·exp(β·[1, ln vrank_{t-1}, ln 前收, ln σ_{t-1}, pick, drop])；只用於參與上限 20%'),
    drift=o5['drift'], odd=o5['odd'], odd_sell_floor0=True,
    zcap=dict(start='2020-10-26', share_by_dec=share_dec, share_default=float(np.median(list(share_dec.values()))), part_main=1.0, part_sens=0.2),
    e2=dict(edges=edges, prem=prem, note='VWAP_0905-0930 ＝ 開盤 × exp(−prem_hat)'),
    crowd=dict(applied=False, reason='VAL 擁擠檢定未通過（M4 不取代）', c_CAL=r3['versions'][0]['params']['crowd_c']),
    sens=dict(M2=dict(model='M2', Y=allv['v2']['params']['k']), M1=dict(model='M1', kappa=allv['v1']['params']['k']),
              M3V_VAL=dict(model='M3V', **{kk: sel['refit_on_VAL_stability'][kk] for kk in ('delta', 'b')}),
              O5=dict(model='O5', note='o5 原式：0.5σ√(Q/OA_o5)，OA_o5 由 o5 model_spec（κ 0.779）')),
    frictions=dict(LIM='開盤價在漲停（買）／跌停（賣）時該檔當日不成交、順延到次日（每日重試）；2015-06-01 前 7%、後 10%',
                   ODD0='零股賣成本下限 0', CASH2='每次換股保留 2% 現金', FEEMIX='2015 年以前用手續費 B（0.6 折），之後 A（0.28 折、最低 20 元）'),
)
json.dump(M, open(C.DATA / 'model_calib.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(M, ensure_ascii=False, indent=1)[:3000])
