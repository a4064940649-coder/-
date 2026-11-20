#!/usr/bin/env python3
"""S83 第 2 步：只讀訓練段（結果視窗最後一個月 ≤ 2015-12）→ 決定每個（指標, 結果）的方向與分三組門檻，寫 PREREG.json＋sha256。
驗證段（2016-01 起）的任何關係在本步都不讀。PREREG.json 已存在就拒絕。python3 -I -B prereg_write.py"""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s83lib as E
HERE = Path(__file__).resolve().parent
assert not (HERE / 'PREREG.json').exists()
I = pd.read_csv(HERE / 'indicators_monthly.csv', index_col=0); O = pd.read_csv(HERE / 'outcomes_monthly.csv', index_col=0)
mon = pd.PeriodIndex(I.index, freq='M'); TE = pd.Period(E.TRAIN_END, 'M')
HZ = dict(sync1=0, sync3=0, lead1=1, lead3=3, lead6=6, sig_lead1=2, sig_lead3=4)   # 結果視窗超出指標月份的月數（次要結果的訊號期約在次月中開始，保守多留 1 個月）
tr_ind = np.asarray(mon <= TE)
thr = {}; train = {}
for i in E.IND:
    x = I.loc[tr_ind, i].dropna()
    thr[i] = dict(lo=float(np.quantile(x, 1 / 3)), hi=float(np.quantile(x, 2 / 3)), n_train=int(len(x)), median=float(x.median()))
pairs = {}
for o in E.SYNC_OUT + E.LEAD_OUT + E.SEC_OUT:
    m = np.asarray(mon + HZ[o] <= TE)
    for i in E.IND:
        if o in E.SYNC_OUT and i not in E.SYNC_IND:
            continue
        x = I.loc[m, i].to_numpy(); y = O.loc[m, o].to_numpy()
        rho = E.spearman(x, y); tm = E.tercile_means(x, y, thr[i]['lo'], thr[i]['hi'])
        pairs[f'{i}|{o}'] = dict(train_rho=rho, sign=int(1 if (rho if np.isfinite(rho) else 0) >= 0 else -1), n=int((np.isfinite(x) & np.isfinite(y)).sum()), train_terciles=tm)
pre = dict(
    task='S83 環境溫度計：策略賺得多的年代能不能事前或同步辨認（不做擇時規則、不改正式策略）',
    written_before_validation=True,
    data='現行帳戶＝S75 V_BRIDGE 標準路徑（100 萬、位移 0），驗收年化 23.52%；結果＝帳戶累積報酬 − 0050 含息累積報酬（日曆月）',
    train='結果視窗最後一個月 ≤ 2015-12', valid='指標月份 ≥ 2016-01（到 2026-10；lead 視窗不足者自然缺值）',
    indicators=[dict(id=a, name=b, definition=c) for a, b, c in E.INDICATORS],
    outcomes=dict(sync1='當月超額', sync3='近 3 個月（含當月）超額', lead1='之後 1 個月超額', lead3='之後 3 個月超額', lead6='之後 6 個月超額',
                  sig_lead1='次要：月底後第一個訊號期 SURGE 前 10 名（無成本）− 0050，持有 1 期', sig_lead3='次要：同上持有 3 期'),
    thresholds=thr, pairs=pairs,
    families=dict(lead=dict(n_tests=len(E.IND) * len(E.LEAD_OUT), desc='15 指標 × 之後 1／3／6 個月'), sync=dict(n_tests=len(E.SYNC_IND) * len(E.SYNC_OUT), desc='14 指標 × 當月／近 3 個月'),
                  secondary=dict(n_tests=len(E.IND) * len(E.SEC_OUT), desc='15 指標 × 純訊號 1／3 期；只描述，不列入過關')),
    test='驗證段：統計量＝訓練段方向 × 驗證段 Spearman；虛無分佈＝把指標整段循環平移 12…n−12 個月（保留自我相關與視窗重疊）；家族內取每次平移的最大值做多重比較校正（FWER）。',
    pass_rule=dict(
        lead='「通過驗證的領先指標」須同時：(a) 驗證段方向與訓練段相同；(b) 領先家族 45 個檢定的校正 p < 0.05；(c) 依訓練段門檻分組，驗證段「高組 − 低組」平均超額方向與訓練段相同，且兩組各 ≥ 12 個月（不足則此條判為無法判定＝不通過）。通過者仍標示「需獨立複製」。',
        sync='「可用的同步指標」須同時：(a) 方向相同；(b) 同步家族 28 個檢定的校正 p < 0.05；(c) 同上分組條件。',
        fallback='若沒有任何領先指標通過：結論寫「沒有領先指標」，只從通過的同步指標中挑 1～3 個當環境溫度計（優先順序：校正 p 小者；彼此驗證段 Spearman > 0.8 者只留一個）；若同步也沒有通過，溫度計只當「描述年代」的讀數（依年代平均差異挑選，並明講沒有通過逐月檢定）。'),
    era_description='另外描述（不檢定）：2008–2012、2013–2015、2016–2018、2019–2023、2024–2026 各指標平均；指標 36 個月移動平均與帳戶 36 個月移動超額的關係。',
    known_before_writing=['S76：全股池「之後 3 期漲超過 50%」比例 2013–2015 1.3%、2016–2018 1.6%、2019–2023 3.1%、2024 起 4.9%；帳戶對 0050 年化超額四小段 +1.2／+3.1／+4.5／+20.4（年代層級的答案已知，驗證段對「年代差異」不是全新資料）',
                          'S59（2011–2026 另一口徑）：7 個環境變數逐月都解釋不了超額；S80：看大盤進出沒用',
                          '本題寫 PREREG 前沒有看過任何指標與 2016 年後超額的逐月關係'],
    not_a_timing_rule=True)
s = json.dumps(pre, ensure_ascii=False, indent=1)
(HERE / 'PREREG.json').write_text(s, encoding='utf-8')
h = hashlib.sha256((HERE / 'PREREG.json').read_bytes()).hexdigest()
(HERE / 'PREREG.sha256').write_text(h + '  PREREG.json\n')
print(h)
for k, v in pairs.items():
    print(k, round(v['train_rho'], 3), v['n'])
