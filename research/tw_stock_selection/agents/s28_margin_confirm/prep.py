#!/usr/bin/env python3
"""S28 第 0 步（不算任何報酬）：面板（除權息修正）、季報毛利率／營業利益率資料涵蓋、PIT 季別對照、
F1 名單上的 RY／DGM／DOPM 分布、各候選規則會改動多少名單（只數檔數）、「由衰轉盛」股（μ ≤ 0）被判定的比例。
輸出 data_check.json。資料不足門檻（事前寫在這裡）：訓練期或驗證期 F1 的 DGM 可用率 < 80%，或 PIT 季別無法決定 → 判資料不足並停。
python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s28lib
from s28lib import s20lib, s10lib, s4lib, stlib, cslib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s28lib.load()
A, _ = s4lib.innov_panel(P)
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U = stlib.universe(P, 'twse') & ~A
del A
G = s20lib.growth_features(P)
RT = dict(S=G['S'], av=G['av'])
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
FQ = s28lib.fin_quarters()
df = s28lib.f1_margin_table(P, U, RT, S0, FQ, G)
df['mper'] = pd.PeriodIndex(df['month'], freq='M')
df['period'] = np.where(df['mper'] <= s28lib.TRAIN_LAST_MONTH, 'train', np.where(df['mper'] >= s28lib.VALID_FIRST_MONTH, 'valid', 'pre'))
df = df[df['mper'] >= pd.Period('2016-01', 'M')]
print('載入', f'{time.time()-t:.0f}s', rss(), 'MB', flush=True)

chk = dict(panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date()), exright_fix=bool(P.exright_fix),
                      exright_info={k: v for k, v in (P.exright_info or {}).items() if not isinstance(v, (list, dict))}))
kqs = FQ['DGM'].columns
chk['fin_source'] = dict(file='agents/o_pead_margin_surprise_low_attention/data/fin_long.parquet（MOPS XBRL 單季財報，公開封存 jb7k6295r7-bot/tw-stock-data；只讀）',
                         first_q=f'{kqs.min()//4}Q{kqs.min()%4+1}', last_q=f'{kqs.max()//4}Q{kqs.max()%4+1}', n_codes=int(len(FQ['DGM'].index)),
                         fields='rev_s 單季營收、gp_s 單季毛利、opi_s 單季營業利益（Q4＝全年減前三季）',
                         dgm_first_q='2014Q1（需要前一年同季）', avail_file='avail_dates.parquet（市場可得日 A；上市 2017-04 起直接觀察證交所 BWIBBU 欄位切換日，之前推斷）')
# PIT 季別：每個訊號列用的「法定期限已過的最新季」
pit = []
for x in S0['info']:
    d = P.IDX[x['k']]
    k0 = s28lib.latest_kq_by_deadline(d)
    pit.append(dict(month=x['month'], sig_date=str(d.date()), kq=f'{k0//4}Q{k0%4+1}', days_after_deadline=int((d - s28lib.deadline(k0 // 4, k0 % 4 + 1)).days)))
pitdf = pd.DataFrame(pit)
chk['pit_rule'] = ('訊號列 k（日期 d）：季 kq 可用 ⇔ 法定期限 deadline(kq) < d 且（市場可得日 A 缺值或 A < d）；取法定期限已過的最新季，'
                   '該檔該季 A ≥ d（晚交）時往回 1 季。法定期限：證交法第 36 條 Q1 5/15、Q2 8/14、Q3 11/14、Q4 次年 3/31（金融保險、-KY 的 Q2 為 8/31，訊號日 9/10 之後，不影響）。'
                   '訊號日次日開盤才進場，所以進場前財報已公告。')
chk['pit_by_signal_month'] = dict(days_after_deadline_min=int(pitdf['days_after_deadline'].min()), days_after_deadline_max=int(pitdf['days_after_deadline'].max()),
                                  by_calendar_month={int(mo): dict(kq_quarter=sorted(set(g['kq'].str[-2:])), days_after_deadline=[int(g['days_after_deadline'].min()), int(g['days_after_deadline'].max())])
                                                     for mo, g in pitdf.assign(mo=pd.to_datetime(pitdf['sig_date']).dt.month).groupby('mo')},
                                  first=pit[0], last=pit[-1])
for per in ('train', 'valid'):
    o = {}
    for grp in ('F1', 'R11_20'):
        d = df[(df['period'] == per) & (df['grp'] == grp)]
        if grp == 'R11_20':
            d = d[d['ret20'] > 0.10]
        k0s = d['k'].map(lambda k: s28lib.latest_kq_by_deadline(P.IDX[k]))
        o[grp] = dict(n=int(len(d)), n_months=int(d['month'].nunique()),
                      kq_found=float((d['kq'] >= 0).mean()), used_fallback_prev_q=float(((d['kq'] >= 0) & (d['kq'] < k0s)).mean()),
                      RY_avail=float(np.isfinite(d['RY']).mean()), DGM_avail=float(np.isfinite(d['DGM']).mean()), DOPM_avail=float(np.isfinite(d['DOPM']).mean()),
                      age_days_median=float(np.nanmedian(d['age_days'])), age_days_max=float(np.nanmax(d['age_days'])),
                      RY_pos=float((d['RY'] > 0).mean()), RY_le0=float((d['RY'] <= 0).mean()),
                      DGM_q=[float(np.nanquantile(d['DGM'], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
                      DOPM_q=[float(np.nanquantile(d['DOPM'], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
                      share_RYpos_DGM_lt_m2=float(((d['RY'] > 0) & (d['DGM'] < -2)).mean()), share_RYpos_DGM_lt_m5=float(((d['RY'] > 0) & (d['DGM'] < -5)).mean()),
                      share_RYle0_DGM_lt_m2=float(((d['RY'] <= 0) & (d['DGM'] < -2)).mean()))
        if grp == 'F1':
            miss = d[~np.isfinite(d['DGM'])]
            o[grp]['DGM_missing_examples'] = miss['code'].value_counts().head(10).to_dict()
            o[grp]['DGM_missing_fin_format'] = {str(FQ['ind'].get(c, '無財報')): int(v) for c, v in miss['code'].map(lambda c: FQ['ind'].get(c, '無財報')).value_counts().items()}
    chk[per] = o
# 規則改動（只數檔數）
eff = {}
cache = {}
for r in s28lib.RULES:
    _, lg = s28lib.build_variant(P, U, RT, S0, FQ, r, cache)
    o = {}
    for per in ('train', 'valid'):
        inper = (lambda m: pd.Period('2016-01', 'M') <= m <= s28lib.TRAIN_LAST_MONTH) if per == 'train' else (lambda m: m >= s28lib.VALID_FIRST_MONTH)
        L = [x for x in lg if inper(pd.Period(x['month'], 'M'))]
        no = sum(len(x['orig']) for x in L); nn = sum(len(x['new']) for x in L)
        rem = sum(len(set(x['orig']) - set(x['new'])) for x in L); add = sum(len(set(x['new']) - set(x['orig'])) for x in L)
        o[per] = dict(n_orig=no, n_new=nn, removed=rem, added=add, months_changed=sum(1 for x in L if x['orig'] != x['new']), n_months=len(L))
    eff[r['id']] = o
chk['rule_effect_preview'] = eff
# 由衰轉盛（μ ≤ 0：過去 24 個月營收年差平均 ≤ 0，S20 的定義）F1 被判定比例（只看特徵，不看報酬）
f1 = df[df['grp'] == 'F1']
tv = {}
for per in ('train', 'valid'):
    d = f1[f1['period'] == per]
    o = {}
    for lab, g in (('turnaround_mu_le0', d[d['MU'] <= 0]), ('mu_gt0', d[d['MU'] > 0]), ('mu_missing', d[~np.isfinite(d['MU'])])):
        o[lab] = dict(n=int(len(g)), RY_le0=float((g['RY'] <= 0).mean()) if len(g) else None,
                      flag_GM2=float(((g['RY'] > 0) & (g['DGM'] < -2)).mean()) if len(g) else None,
                      flag_GM5=float(((g['RY'] > 0) & (g['DGM'] < -5)).mean()) if len(g) else None,
                      DGM_lt_m2_any_RY=float((g['DGM'] < -2).mean()) if len(g) else None,
                      DGM_median=float(np.nanmedian(g['DGM'])) if len(g) and np.isfinite(g['DGM']).any() else None)
    tv[per] = o
chk['turnaround_check'] = tv
# A 與法定期限的關係（上市非金融、2017Q1 起直接觀察）：A ≤ 訊號日的比例，用 F1 上的 (code, kq)
ok = []
for r in f1.itertuples():
    if r.kq < 0:
        continue
    a = FQ['A'].get((r.code, r.kq))
    ok.append(dict(period=r.period, has_A=a is not None, A_lt_sig=(a is not None and a < P.IDX[r.k])))
ok = pd.DataFrame(ok)
chk['A_on_used_quarters'] = {per: dict(has_A=float(g['has_A'].mean()), A_before_signal=float(g['A_lt_sig'][g['has_A']].mean())) for per, g in ok.groupby('period')}
# 資料不足判定
suff = all(chk[per]['F1']['DGM_avail'] >= 0.80 for per in ('train', 'valid'))
chk['sufficiency'] = dict(rule='訓練期與驗證期 F1 的 DGM 可用率都 ≥ 80%，且每個訊號列都能決定 PIT 季別', DGM_avail_train=chk['train']['F1']['DGM_avail'],
                          DGM_avail_valid=chk['valid']['F1']['DGM_avail'], pit_ok=True, verdict='資料足夠' if suff else '資料不足')
chk['runtime_s'] = round(time.time() - t, 1); chk['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
print(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
