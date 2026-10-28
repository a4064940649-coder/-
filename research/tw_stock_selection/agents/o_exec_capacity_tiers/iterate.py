#!/usr/bin/env python3
"""迭代建模（只有執行量模型；不看任何策略報酬）→ iterations.json、data/model_spec.json
事前寫定（本檔寫成後才第一次執行）：
  版本 v1～v5 同時定義三個元件：
    OA（開盤集合競價量）：ln(r_bar) 對變數 OLS，r_bar＝Yahoo 09:05–09:10 量 ÷ ADV60（官方口徑）；OA÷V60＝κ×exp(預測)，κ＝vol_curve 的 OA_central÷09:05–09:10 量中位數
      v1 const｜v2 +lnrank+lnprice+lnsig｜v3 v2+absgap｜v4 v3+pick+drop（換股日入選／剔除旗標）｜v5 v4 但 absgap 拆成 gapup、gapdn
    drift（9:10 價÷開盤−1）：v1 單一平均｜v2、v3 依跳空 3 桶（<−1%、−1～1%、>1%）｜v4、v5 依跳空 6 桶（−3%、−1%、0、1%、3%）
    odd（零股買／賣成本）：v1 單一中位數｜v2、v3 依價位 4 桶（50、100、300）｜v4、v5 價位 4 桶×名次 2 桶（150）
  樣本切分：5 分 K 2024-02-15～2025-06-30 擬合、2025-07-01～2026-04-01 檢驗；零股 2026-07-13～08-31 擬合、09-01～10-27 檢驗。只用前 300 名池。
  損失：OA＝檢驗期 換股日情境列（pick 或 drop）的 log RMSE（主），全部列 log RMSE（副）；drift＝檢驗期 MSE；odd＝檢驗期 買、賣成本 MAE 平均。
  選版：score＝Σ 三元件（損失 ÷ v1 損失）；最小者；差 < 0.01 取較簡單者（版本號小）。選定後以全部校準資料重新擬合係數。
"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import xlib as X

VERS = {
    'v1': dict(oa=['const'], drift=[], odd_p=[], odd_r=[]),
    'v2': dict(oa=['const', 'lnrank', 'lnprice', 'lnsig'], drift=[-0.01, 0.01], odd_p=[50, 100, 300], odd_r=[]),
    'v3': dict(oa=['const', 'lnrank', 'lnprice', 'lnsig', 'absgap'], drift=[-0.01, 0.01], odd_p=[50, 100, 300], odd_r=[]),
    'v4': dict(oa=['const', 'lnrank', 'lnprice', 'lnsig', 'absgap', 'pick', 'drop'], drift=[-0.03, -0.01, 0.0, 0.01, 0.03], odd_p=[50, 100, 300], odd_r=[150]),
    'v5': dict(oa=['const', 'lnrank', 'lnprice', 'lnsig', 'gapup', 'gapdn', 'pick', 'drop'], drift=[-0.03, -0.01, 0.0, 0.01, 0.03], odd_p=[50, 100, 300], odd_r=[150]),
}
SPLIT_5M = pd.Timestamp('2025-07-01'); SPLIT_ODD = pd.Timestamp('2026-09-01')

Y = X.load_5m()
P = pd.concat([pd.read_csv(HERE / 'data' / 'plans_ab.csv', dtype=str), pd.read_csv(HERE / 'data' / 'plan_v4.csv', dtype=str)])
pick, drop = set(), set()
for s, g in P.groupby('seed'):
    prev = set()
    for e, pk in zip(g['exec'], g.picks):
        cur = set(pk.split(',')); d = pd.Timestamp(e)
        pick |= {(d, c) for c in cur}; drop |= {(d, c) for c in prev - cur}; prev = cur
Y['pick'] = [float((d, c) in pick) for d, c in zip(Y.date, Y.code)]
Y['dropd'] = [float((d, c) in drop) for d, c in zip(Y.date, Y.code)]
Y = Y[Y.vrank_l1.notna() & (Y.vrank_l1 <= 300) & np.isfinite(Y.gap) & np.isfinite(Y.sig_l1) & (Y.prevc > 0)].copy()
Y['ctx'] = (Y['pick'] + Y['dropd']) > 0

def design(df, vars_):
    cols = {'const': np.ones(len(df)), 'lnrank': np.log(df.vrank_l1.clip(lower=1)), 'lnprice': np.log(df.prevc), 'lnsig': np.log(df.sig_l1.clip(lower=1e-4)),
            'absgap': df.gap.abs(), 'gapup': df.gap.clip(lower=0), 'gapdn': (-df.gap).clip(lower=0), 'pick': df['pick'], 'drop': df['dropd']}
    return np.column_stack([np.asarray(cols[v], float) for v in vars_])

YA = Y[np.isfinite(Y.r_bar) & (Y.r_bar > 0)].copy(); YA['lr'] = np.log(YA.r_bar)
YD = Y[np.isfinite(Y.drift910)].copy()
O = X.load_odd(); O = O[(O.close >= 10) & O.vrank_off.notna() & (O.vrank_off <= 300)].copy()
V = X.load_vc()
kc = (V.OA_central / V.bar0905_eq).replace([np.inf, -np.inf], np.nan).dropna()
KAPPA = float(kc.median()); KAPPA_LO = float(kc.quantile(0.25)); KAPPA_HI = float((V.OA_upper / V.bar0905_eq).replace([np.inf, -np.inf], np.nan).dropna().median())

def fit_oa(df, vars_):
    Xm = design(df, vars_); b, *_ = np.linalg.lstsq(Xm, df.lr.to_numpy(), rcond=None); return b
def bucket_mean(x, y, edges):
    k = np.searchsorted(np.array(edges), x, side='right') if edges else np.zeros(len(x), int)
    m = pd.Series(y).groupby(k).mean(); out = np.array([m.get(i, np.nan) for i in range(len(edges) + 1)])
    return np.where(np.isfinite(out), out, np.nanmean(y))
def bucket_med2(pr, rk, y, pe, re_):
    i = np.searchsorted(np.array(pe), pr, side='right') if pe else np.zeros(len(pr), int)
    j = np.searchsorted(np.array(re_), rk, side='right') if re_ else np.zeros(len(pr), int)
    g = pd.Series(y).groupby([i, j]).median(); tot = float(np.nanmedian(y))
    M = np.full((len(pe) + 1, len(re_) + 1), tot)
    for (a, b), v in g.items():
        if np.isfinite(v): M[a, b] = v
    return M
def apply2(M, pr, rk, pe, re_):
    i = np.searchsorted(np.array(pe), pr, side='right') if pe else np.zeros(len(pr), int)
    j = np.searchsorted(np.array(re_), rk, side='right') if re_ else np.zeros(len(pr), int)
    return M[i, j]

res = {}
tr5, te5 = YA[YA.date < SPLIT_5M], YA[YA.date >= SPLIT_5M]
trd, ted = YD[YD.date < SPLIT_5M], YD[YD.date >= SPLIT_5M]
tro, teo = O[O.date < SPLIT_ODD], O[O.date >= SPLIT_ODD]
for v, spec in VERS.items():
    b = fit_oa(tr5, spec['oa'])
    pred = design(te5, spec['oa']) @ b
    err = te5.lr.to_numpy() - pred
    ctx = te5.ctx.to_numpy()
    oa = {'rmse_ctx': float(np.sqrt(np.mean(err[ctx] ** 2))), 'rmse_all': float(np.sqrt(np.mean(err ** 2))), 'bias_ctx': float(err[ctx].mean()),
          'n_test': int(len(te5)), 'n_test_ctx': int(ctx.sum()), 'n_train': int(len(tr5)), 'coef_train': dict(zip(spec['oa'], map(float, b)))}
    dm = bucket_mean(trd.gap.to_numpy(), trd.drift910.to_numpy(), spec['drift'])
    k = np.searchsorted(np.array(spec['drift']), ted.gap.to_numpy(), side='right') if spec['drift'] else np.zeros(len(ted), int)
    dr = {'mse': float(np.mean((ted.drift910.to_numpy() - dm[k]) ** 2)), 'vals_train': list(map(float, dm)),
          'mse_ctx': float(np.mean(((ted.drift910 - dm[k])[ted.ctx.to_numpy()]) ** 2))}
    l = []
    od = {}
    for side, col in (('buy', 'buy_cost'), ('sell', 'sell_cost')):
        a, t_ = tro[np.isfinite(tro[col])], teo[np.isfinite(teo[col])]
        M = bucket_med2(a.close.to_numpy(), a.vrank_off.to_numpy(), a[col].to_numpy(), spec['odd_p'], spec['odd_r'])
        p = apply2(M, t_.close.to_numpy(), t_.vrank_off.to_numpy(), spec['odd_p'], spec['odd_r'])
        mae = float(np.mean(np.abs(t_[col].to_numpy() - p))); l.append(mae); od[f'mae_{side}'] = mae; od[f'M_{side}_train'] = M.tolist()
    od['mae'] = float(np.mean(l))
    res[v] = {'spec': spec, 'oa': oa, 'drift': dr, 'odd': od}
base = res['v1']
for v in res:
    res[v]['score'] = res[v]['oa']['rmse_ctx'] / base['oa']['rmse_ctx'] + res[v]['drift']['mse'] / base['drift']['mse'] + res[v]['odd']['mae'] / base['odd']['mae']
order = sorted(res, key=lambda v: (res[v]['score'], v))
best = order[0]
for v in sorted(res):
    if res[v]['score'] - res[best]['score'] < 0.01 and v < best: best = v
sp = VERS[best]
# 全部校準資料重新擬合
b = fit_oa(YA, sp['oa'])
dm = bucket_mean(YD.gap.to_numpy(), YD.drift910.to_numpy(), sp['drift'])
Mb = bucket_med2(O.close.to_numpy(), O.vrank_off.to_numpy(), O.buy_cost.to_numpy(), sp['odd_p'], sp['odd_r'])
Ms = bucket_med2(O.close.to_numpy(), O.vrank_off.to_numpy(), O.sell_cost.to_numpy(), sp['odd_p'], sp['odd_r'])
resid = YA.lr.to_numpy() - design(YA, sp['oa']) @ b
model = {'version': best, 'oa': {'vars': sp['oa'], 'coef': list(map(float, b)), 'kappa': KAPPA, 'kappa_lo_p25': KAPPA_LO, 'kappa_hi_upper': KAPPA_HI,
                                 'resid_sd_log': float(resid.std()), 'resid_sd_log_ctx': float(resid[YA.ctx.to_numpy()].std()), 'n_fit': int(len(YA))},
         'drift': {'edges': sp['drift'], 'vals': list(map(float, dm)), 'n_fit': int(len(YD))},
         'odd': {'price_edges': list(map(float, sp['odd_p'])), 'rank_edges': list(map(float, sp['odd_r'])), 'buy': Mb.tolist(), 'sell': Ms.tolist(), 'n_fit': int(len(O))},
         'yahoo_to_official': X.YAHOO_TO_OFFICIAL,
         'note': 'OA÷V60＝κ×exp(coef·x)；x 的 pick／drop 只在換股日（入選含續抱＝pick，剔除＝drop）為 1，順延日為 0。drift：9:10 價÷開盤−1 依跳空分桶平均。odd：零股買／賣成本（相對同時點整股價）分桶中位數。'}
it = {'prereg_rules': __doc__, 'versions': res, 'selection_order': order, 'selected': best, 'n_versions': len(VERS),
      'final_model': model, 'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
json.dump(it, open(HERE / 'iterations.json', 'w'), ensure_ascii=False, indent=1, default=float)
json.dump(model, open(HERE / 'data' / 'model_spec.json', 'w'), ensure_ascii=False, indent=1, default=float)
for v in sorted(res):
    print(v, round(res[v]['score'], 4), 'oa', round(res[v]['oa']['rmse_ctx'], 4), round(res[v]['oa']['rmse_all'], 4), round(res[v]['oa']['bias_ctx'], 3),
          'drift', res[v]['drift']['mse'], 'odd', round(res[v]['odd']['mae'], 6))
print('selected', best, json.dumps(model, ensure_ascii=False, default=float)[:1500])
