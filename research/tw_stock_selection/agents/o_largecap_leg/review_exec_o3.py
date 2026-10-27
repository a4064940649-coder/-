#!/usr/bin/env python3
"""反方審查（成本與可成交）：只讀研究員檔案與 lcore，不寫任何研究員輸出（只印到 stdout）。
檢查：
 (1) 換股日開盤「跌停」要賣的部位（實務上可能賣不掉）在 L 腿與 MAIN 的頻率與次日損益；
 (2) 開盤漲停遞補（需要 9:00 盯盤看開盤，再下第二張單）的頻率；遞補股當天開盤→收盤漂移；
 (3) 2021-09 以前處置（V3 近似全期名單）在 L 腿入選中的比例，及萬海／長榮／創意等主要貢獻股是否落在處置期；
 (4) 2020-10-26 盤中零股上線以前：30 萬／100 萬帳戶只能整張或盤後零股（收盤價）→ 以「換股日收盤成交」重算 L 腿對池等權；
     整張限制下 L 腿實際可買到的比例；
 (5) 成本拖累：L 腿每月單邊周轉 × 來回成本，對照訊號強度；
 (6) 開盤競價容量：每檔下單金額 ÷ 換股日開盤成交額的代理（以 60 日均成交值 × 5% 近似開盤占比）。"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcore as L
sys.path.pop(0)
import numpy as np, pandas as pd
K, V, O = L.K, L.V, L.O
IDX, COLS = L.IDX, L.COLS
t1 = L.T1

def tick(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def limit_dn(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    ref = K.ref_price(t); raw = ref * (1 - lim); tk = tick(raw)
    ld = np.ceil(raw / tk - 1e-9) * tk
    return np.isfinite(K.On[t]) & np.isfinite(ref) & (K.On[t] <= ld + 1e-6)

leg = L.plan_leg(50, 5, '55', 'ew')
mp = L.MAIN_PLAN
SPLIT = L.SPLIT

def per(lab, sd): return 'SEL' if sd < SPLIT else ('HOLD' if sd >= pd.Timestamp('2024-01-01') else 'VAL_A')

# ---------- (1) 換股日開盤跌停要賣 ----------
def sells_at_limitdown(plan, getpk):
    out = {'SEL': [0, 0], 'VAL_A': [0, 0], 'HOLD': [0, 0]}; loss = []
    prev = set()
    for p in plan:
        e = p[0]; pk = set(np.asarray(getpk(p), int).tolist()); sd = p[3]
        sells = prev - pk
        if sells:
            ld = limit_dn(e)
            for j in sells:
                k = per('', sd); out[k][0] += 1
                if ld[j]:
                    out[k][1] += 1
                    # 若賣不掉、改次日開盤賣：額外損益＝(次日開盤 / 當日開盤) − 1（以還原報酬近似）
                    if e + 1 < len(IDX) and np.isfinite(K.On[e + 1, j]) and np.isfinite(K.Cn[e, j]) and np.isfinite(K.Cn[e + 1, j]):
                        r = (K.Cn[e, j] / K.On[e, j]) * K.GROW[e + 1, j] * (K.On[e + 1, j] / K.Cn[e + 1, j]) - 1
                        loss.append((str(sd.date()), COLS[j], round(float(r) * 100, 2)))
        prev = pk
    return out, loss
r1L, lossL = sells_at_limitdown(leg, lambda p: p[1])
r1M, lossM = sells_at_limitdown(mp, lambda p: p[2])
print('(1) 換股日開盤跌停要賣 [賣出筆數, 開盤跌停筆數]  L:', r1L, ' MAIN:', r1M)
print('    L 腿跌停賣出改次日開盤的額外報酬(%):', lossL[:20])

# ---------- (2) 開盤漲停遞補 ----------
subs = {'SEL': [0, 0], 'VAL_A': [0, 0], 'HOLD': [0, 0]}; sub_drift = []
held = set()
for e0, d in zip(L.EXEC, L.SIG):
    cs = L.large_cs(d, 50, '55'); e = cs['e']
    gu = K.gap_up_open(e)
    cand = V.V2.order(cs, 'official')[:len(cs['pool'])]
    pick = []; skipped = 0
    for j in cand:
        if gu[j] and j not in held: skipped += 1; continue
        pick.append(j)
        if len(pick) == 5: break
    k = per('', d); subs[k][0] += 1; subs[k][1] += int(skipped > 0)
    held = set(pick)
print('(2) 換股月數、需遞補（開盤漲停跳過）的月數:', subs)

# 換股日買進股的開盤→收盤漂移（若只能盤後零股以收盤價成交，損失多少）
def oc_drift(plan, getpk, getw=None):
    res = {'SEL': [], 'VAL_A': [], 'HOLD': []}
    prev = set()
    for p in plan:
        e = p[0]; pk = np.asarray(getpk(p), int); sd = p[3]
        buys = [j for j in pk if j not in prev]; sells = [j for j in prev if j not in set(pk.tolist())]
        tr = K.TRADED[e] & np.isfinite(K.On[e])
        def oc(j): return K.Cn[e, j] / K.On[e, j] - 1 if tr[j] else np.nan
        b = np.nanmean([oc(j) for j in buys]) if buys else 0.0
        s = np.nanmean([oc(j) for j in sells]) if sells else 0.0
        nb = len(buys) / max(len(pk), 1); ns = len(sells) / max(len(pk), 1)
        # 以收盤成交而非開盤：買進付出 +oc、賣出拿到 +oc；淨影響 ≈ −nb·b + ns·s（占腿市值）
        res[per('', sd)].append(-nb * b + ns * s)
        prev = set(pk.tolist())
    return {k: (round(float(np.nanmean(v)) * 100, 3), round(float(K.tstat(pd.Series(v).dropna())), 2), len(v)) for k, v in res.items()}
print('(4a) 改用換股日收盤成交（盤後零股）每月淨影響 pp（均值, t, n）  L:', oc_drift(leg, lambda p: p[1]), ' MAIN:', oc_drift(mp, lambda p: p[2]))

# ---------- (3) 處置（V3 近似全期名單） ----------
try:
    V3D = HERE.parent / 'o_infra_v3_rebuild'
    sys.path.insert(0, str(V3D)); import o3_core as O3; sys.path.pop(0)
    H = O3.load_disposal_hist()
    def inH(d, j):
        c = COLS[j]; x = H[(H['code'] == c) & (H['start'] <= d) & (H['end'] >= d)]
        return len(x) > 0
    def inH_window(d0, d1, j):
        c = COLS[j]; x = H[(H['code'] == c) & (H['start'] <= d1) & (H['end'] >= d0)]
        return len(x) > 0
    hits = {'SEL': [0, 0], 'VAL_pre2109': [0, 0]}; hit_list = []
    for i, p in enumerate(leg):
        sd = p[3]
        if sd >= pd.Timestamp('2021-09-24'): continue
        k = 'SEL' if sd < SPLIT else 'VAL_pre2109'
        nxt = IDX[leg[i + 1][0]] if i + 1 < len(leg) else IDX[t1]
        for j in p[1]:
            hits[k][0] += 1
            if inH(sd, j) or inH(IDX[p[0]], j): hits[k][1] += 1; hit_list.append((str(sd.date()), COLS[j], 'sig/exec'))
            elif inH_window(IDX[p[0]], nxt, j): hit_list.append((str(sd.date()), COLS[j], 'during_hold'))
    print('(3) 2021-09 以前 L 腿入選 [名額, 訊號／成交日在處置]:', hits)
    print('    明細（含持有期間進處置，需預收款券才能加碼／賣出不受限）:', [h for h in hit_list if h[1] in ('2615', '2603', '2609', '3443', '2059', '2368')][:30], ' 總數', len(hit_list))
except Exception as ex:
    print('(3) V3 處置名單讀取失敗:', repr(ex))

# 2021 年航運股在 L 腿的月份與當期報酬
per_ret = L.leg_period_returns(leg, t1)
ship = []
for d, pk, w, r, pool, rp in per_ret:
    if d.year == 2021:
        for j, rj in zip(pk, r):
            if COLS[j] in ('2615', '2603', '2609'):
                ship.append((str(d.date()), COLS[j], round(float(rj) * 100, 1), round(float(rp.mean()) * 100, 1)))
print('    2021 航運股入選（訊號日, 代碼, 期報酬%, 池等權%）:', ship)

# ---------- (4b) 整張限制（2020-10-26 以前沒有盤中零股） ----------
for cap in (3e5, 1e6, 1e7):
    per_name = cap * 0.3 / 5
    rows = []
    for p in leg:
        e = p[0]
        if IDX[e] >= pd.Timestamp('2020-10-26'): continue
        px = K.On[e, p[1]] * 1000
        rows.append(np.mean(px <= per_name))
    print(f'(4b) 資金 {cap/1e4:.0f} 萬：2020-10 前 L 腿每檔 {per_name/1e4:.1f} 萬，買得起一整張的名額比例 {np.mean(rows):.3f}')

# ---------- (5) 成本拖累 ----------
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['RESULTS']
tov = R['turnover']['L_ALONE']
rt_n = (L.NORMAL[0] + L.NORMAL[1]); rt_s = (L.STRESS3[0] + L.STRESS3[1])
print(f'(5) L 腿單邊周轉 {tov:.3f}/月；一般來回 {rt_n*100:.3f}% → 拖累 {tov*rt_n*100:.3f}pp/月；stress3 來回 {rt_s*100:.3f}% → {tov*rt_s*100:.3f}pp/月')
for p in ('SEL', 'VAL', 'HOLD'):
    print('    L vs 池等權', p, {c: R['leg'][p][f'L_vs_poolEW_{c}'] for c in ('normal', 'stress3', 'stress_old')})

# ---------- (6) 容量 ----------
for cap in (1e7, 3e7):
    per_name = cap * 0.3 / 5; part = []
    for p in leg:
        v60 = L.large_cs(p[3], 50, '55')['v60'][p[1]]
        part.append(np.max(per_name / (0.05 * v60)))
    print(f'(6) 資金 {cap/1e4:.0f} 萬：每檔 {per_name/1e4:.0f} 萬 ÷（60 日均值×5%）的每月最大值：中位 {np.median(part):.3f}、最大 {np.max(part):.3f}')
