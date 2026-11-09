#!/usr/bin/env python3
"""凍結前煙霧測試（紙上 ia）：
 (1) patch flag 關閉：C0 名單與原 ia_core 逐筆相同、淨值對 main_v2/navs_gapfix 逐日差 0（一般／壓力）；
 (2) 聯集版（patch DISP_OFFICIAL='union'）重現 integrate_o17 C_DSP：名單異動槽數＝15、全期年化 27.6064%（差 ≤ 0.01pp）；
 (3) 處置槽計數（不看報酬）：C0 原名單落在官方處置期的槽（訊號日／成交日，2014-01-01～2021-09-23）與 o17 題 1 逐槽相同；替換版名單異動槽；
 (4) C0 不持有任何 v3 事件代碼於其 lt（C0 Δ＝0 的結構條件）。
不計算替換版、v3 的任何報酬。python3 -I -B s7_smoke_ia.py"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import ia_lib as IL; sys.path.pop(0)
AG = IL.AG
P, CE = IL.load(patched=True); K = P.K; COLS = P.COLS
out = dict(utc=pd.Timestamp.now('UTC').isoformat(), ia_core_sha=IL.IA_SHA, patch_sha=IL.sha(IL.PATCH))
B = {m: IL.with_mask(P, m, lambda: IL.c0_plan(P)) for m in (None, 'union', 'replace')}
PK = {m: {p[3]: [COLS[j] for j in p[1]] for p in B[m]['plan']} for m in B}
L0 = json.loads((AG / 'o_g8_protocol_fin_erratum' / 'ERRATUM.json').read_text(encoding='utf-8'))['lists_C0']
assert all(PK[None][pd.Timestamp(d)] == v for d, v in L0.items() if pd.Timestamp(d) in PK[None])
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
OFF = pd.read_parquet(AG / 'main_v2' / 'navs_gapfix.parquet')
d0 = {}
for c, (bb, ss) in COST.items():
    nav = K.simulate([(p[0], p[1]) for p in B[None]['plan']], bb, ss, B[None]['t0'], B[None]['t1'])[0]
    ref = OFF[f'EXFIN|n10|{c}'].dropna()
    d0[c] = float(np.nanmax(np.abs(nav.reindex(ref.index).to_numpy() - ref.to_numpy())))
assert max(d0.values()) < 1e-9, d0
out['flag_off_vs_main_v2_maxabs'] = d0
def changed(a, b, lo='2006-01-01', hi='2026-12-31'):
    rows = []
    for d in PK[a]:
        if pd.Timestamp(lo) <= d <= pd.Timestamp(hi):
            for c in sorted(set(PK[a][d]) - set(PK[b][d])): rows.append(('out', str(d.date()), c))
            for c in sorted(set(PK[b][d]) - set(PK[a][d])): rows.append(('in', str(d.date()), c))
    return rows
cu = changed(None, 'union'); cr = changed(None, 'replace')
out['union_changed'] = cu; out['replace_changed'] = cr
out['n_union_out'] = sum(1 for x in cu if x[0] == 'out'); out['n_replace_out'] = sum(1 for x in cr if x[0] == 'out')
J17 = json.loads((AG / 'integrate_o17' / 'results.json').read_text(encoding='utf-8'))
assert out['n_union_out'] == J17['RESULTS']['checks']['C_DSP_changed_slots']['all'], (out['n_union_out'], J17['RESULTS']['checks']['C_DSP_changed_slots'])
# (2) 重現 C_DSP 年化
W = {}
for m in (None, 'union'):
    nav = K.simulate([(p[0], p[1]) for p in B[m]['plan']], *COST['normal'], B[m]['t0'], B[m]['t1'])[0]
    W[m] = IL.win(nav, B[m]['sd'])
st = IL.stats_one(W['union'].r.to_numpy(), W['union'].r50.to_numpy(), W[None].r.to_numpy(), W['union'].index, W['union'].yr.to_numpy())
ref = J17['RESULTS']['stats']['C_DSP|normal|freeze|vs_C0']
out['C_DSP_reproduce'] = dict(ann=st['ALL']['ann'], ref=ref['ALL']['ann'], diff=st['ALL']['ann'] - ref['ALL']['ann'], seg3_gex=st['SEG3']['gex'], ref_seg3=ref['SEG3']['gex'])
assert abs(out['C_DSP_reproduce']['diff']) <= 0.01
# (3) 處置槽計數
sys.path.insert(0, str(HERE)); import dispmask_official as DO; sys.path.pop(0)
IDX = P.IDX
def inoff(d, c, which):
    t = IDX.get_loc(d); e = IDX[t + 1]; g = DO.DU[DO.DU.code == c]
    if which == 'sig': return bool(((g.start <= d) & (g.end >= d)).any())
    return bool(((g.start <= e) & (g.end >= e)).any())
R1 = json.loads((AG / 'o_rev_avail_ds_pit_count' / 'results.json').read_text(encoding='utf-8'))['ERRATUM']['R3']['counts']['2014_2021-09-23']
for w in ('sig', 'trd'):
    hit = sorted(f'{d.date()} {c}' for d in PK[None] if pd.Timestamp('2014-01-01') <= d <= pd.Timestamp('2021-09-23') for c in PK[None][d] if inoff(d, c, 'sig' if w == 'sig' else 'trd'))
    out[f'C0_slots_in_official_{w}_2014_2021'] = hit
    assert hit == sorted(R1[f'd_{w}']['slots']), (w, hit)
# 2014～2021 列中 overlap_proxy 比例（P5 描述）
U = DO.DU; m = (U.start >= '2014-01-01') & (U.start <= '2021-09-23')
out['overlap_proxy_share_2014_2021'] = dict(n=int(m.sum()), n_true=int(U.loc[m, 'overlap_proxy'].sum()), share=float(U.loc[m, 'overlap_proxy'].mean()),
                                            note='overlap_proxy＝與 disposal_bear_longhistory 代理事件期間重疊；起訖日全部來自官方公告，不是代理推得')
hitall = [x for x in out['C0_slots_in_official_sig_2014_2021']]
out['C0_slots_overlap_proxy'] = {s: bool(any(((U.code == s.split()[1]) & (U.start <= pd.Timestamp(s.split()[0]) + pd.Timedelta(days=3)) & (U.end >= pd.Timestamp(s.split()[0])) & U.overlap_proxy))) for s in hitall}
# (4) C0 持有 v3 事件代碼
E3 = pd.read_csv(HERE / 'events_v3.csv', dtype={'code': str})
E2 = pd.read_csv(AG / 'o_engine_delist_consideration' / 'events_v2.csv', dtype={'code': str})
codes = set(E3.code) | set(E2.code)
out['C0_slots_in_event_codes'] = sorted(f'{d.date()} {c}' for d in PK[None] for c in PK[None][d] if c in codes)
(HERE / 'smoke_ia.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k not in ('union_changed', 'replace_changed')}, ensure_ascii=False, indent=1, default=str))
print('union_changed', cu); print('replace_changed', cr)
