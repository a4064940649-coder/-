#!/usr/bin/env python3
"""第 4 步：依凍結的 GATE_SPEC 計算三種資料的閘門、選版、判定。寫 data/gate.json。"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, glib
lock = json.load(open(HERE / 'data' / 'gate_spec_lock.json'))
assert hashlib.sha256(open(HERE / 'GATE_SPEC.json', 'rb').read()).hexdigest() == lock['sha256'], 'GATE_SPEC 被改動'
BUILD = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/opm2/build')
L, V, FD, LD = glib.denominator()
NEW = glib.new_listing_mask(L, FD, 130)
P300 = glib.pool300(L, V)
DEL = LD < pd.Timestamp('2026-10-01')
cons = json.load(open(HERE / 'data' / 'consistency.json'))
TH = 0.95
def cov(core, den):
    den = den.reindex(index=core.index, columns=core.columns).fillna(False).values
    hit = core.notna().values & den
    return hit, den
def segstats(core, den):
    hit, d = cov(core, den)
    idx = core.index; out = {}
    for s, (a, b) in glib.SEG.items():
        m = (idx >= a) & (idx <= b)
        out[s] = round(float(hit[m].sum() / max(d[m].sum(), 1)), 4)
    yr = {}
    for y in sorted(set(idx.year)):
        m = idx.year == y; yr[int(y)] = round(float(hit[m].sum() / max(d[m].sum(), 1)), 4)
    return out, yr, int(d.sum())
def evaluate(core, den_base, name, date_basis, identity_ok, overlap_ok, range_ok):
    full, fy, n = segstats(core, den_base)
    pool, py, _ = segstats(core, den_base & P300)
    dl, dly, nd = segstats(core, den_base & pd.DataFrame(np.broadcast_to(DEL.reindex(L.columns).values, L.shape), index=L.index, columns=L.columns))
    sv, svy, _ = segstats(core, den_base & ~pd.DataFrame(np.broadcast_to(DEL.reindex(L.columns).values, L.shape), index=L.index, columns=L.columns))
    seg = {}
    for s in glib.SEG:
        seg[s] = {'G_PIT': date_basis >= 0.98, 'G_SURV': dl[s] >= TH, 'G_COV': full[s] >= TH and pool[s] >= TH,
                  'cov_full': full[s], 'cov_pool300': pool[s], 'cov_delisted': dl[s], 'cov_survivors': sv[s]}
    consist = bool(identity_ok and overlap_ok and range_ok)
    npass = sum(int(seg[s][g]) for s in seg for g in ['G_PIT', 'G_SURV', 'G_COV']) + int(consist)
    segok = [s for s in seg if seg[s]['G_PIT'] and seg[s]['G_SURV'] and seg[s]['G_COV']]
    verdict = '通過' if (len(segok) == 3 and consist) else ('部分' if segok else '不通過')
    return {'name': name, 'segments': seg, 'yearly_full': fy, 'yearly_pool300': py, 'yearly_delisted': dly, 'denominator_stockdays': n, 'delisted_stockdays': nd,
            'date_basis_rate': date_basis, 'CONSIST': {'identity': identity_ok, 'local_overlap': overlap_ok, 'range': range_ok, 'pass': consist},
            'n_pass_of_10': npass, 'segments_ok': segok, 'verdict_raw': verdict}
res = {'gate_spec_sha256': lock['sha256']}
# ---------- MARGN ----------
den_m = L & ~NEW
ident_v1 = cons['TEJ_ineq_dbal_le_buy_minus_sell']['rate'] >= 0.999 and cons['BB_prev_continuity']['rate'] >= 0.999
ident_v23 = ident_v1 and cons['FinBrain_vs_TEJ_delta']['rate'] >= 0.999
overlap = all(cons[k]['rate'] >= 0.99 for k in ['TEJ_vs_BlackBoxOP_level', 'joeyc_vs_BlackBoxOP_level', 'joeyc_vs_margin_hist_level', 'BlackBoxOP_vs_margin_hist_level'])
mv = {}
for v in ['V1', 'V2', 'V3']:
    core = pd.read_parquet(BUILD / f'margn_{v}.parquet').reindex(index=L.index, columns=L.columns)
    mv[v] = evaluate(core, den_m, f'MARGN_{v}', 1.0, ident_v1 if v == 'V1' else ident_v23, overlap, cons[f'range_nonneg_{v}'] >= 1.0)
    # 不扣新上市期的版本（描述）
    full_nx, _, _ = segstats(core, L)
    mv[v]['cov_full_no_newlist_exclusion'] = full_nx
order = {'V1': 1, 'V2': 2, 'V3': 3}
best = sorted(mv, key=lambda v: (-mv[v]['n_pass_of_10'], order[v]))[0]
# 原始來源組成（V1）：各段由哪個來源提供
rs = pd.read_parquet(BUILD / 'margn_rawsrc.parquet').reindex(index=L.index, columns=L.columns)
comp = {}
for s, (a, b) in glib.SEG.items():
    m = (rs.index >= a) & (rs.index <= b); x = rs.values[m][den_m.values[m]]
    comp[s] = {nm: round(float((x == k).mean()), 4) for k, nm in [(1, 'joeyc_official_monthly'), (2, 'BlackBoxOP'), (3, 'margin_hist'), (4, 'TEJ_survivors'), (0, 'missing')]}
res['MARGN'] = {'versions': mv, 'selected': best, 'verdict': mv[best]['verdict_raw'], 'v1_source_mix': comp}
# ---------- QFIIS（日頻主判定＋月頻形成日變體）----------
Q = pd.read_parquet(BUILD / 'qfiis_daily.parquet').reindex(index=L.index, columns=L.columns)
qid = cons['joeyc_vs_voidful_qfii']['ratio_eq_rate'] is not None and cons['joeyc_vs_voidful_qfii']['ratio_eq_rate'] >= 0.99
qr = cons['qfii_range']['pct_in_0_100'] >= 1.0 and cons['qfii_range']['held_le_issued'] >= 1.0
res['QFIIS'] = {'daily': evaluate(Q, L, 'QFIIS_daily', 1.0, True, qid, qr)}
jq = glib.load_joeyc('qfii-twse'); fdates = sorted(set(jq.date))
fd_idx = pd.DatetimeIndex(fdates)
Qm = Q.loc[Q.index.intersection(fd_idx)]
# 月頻：分母＝所有月份（2006-01～2026-10）每月一個形成日；沒有形成日的月份分母照算、分子為 0
months = pd.period_range('2006-01', '2026-10', freq='M')
mrows = []
for p in months:
    ds = [d for d in fdates if d.to_period('M') == p]
    if ds:
        d = ds[0]; den = L.loc[d] if d in L.index else None
        if den is None: continue
        hit = Q.loc[d].notna() & den; mrows.append((p, int(den.sum()), int(hit.sum()), int((den & P300.loc[d]).sum()), int((hit & P300.loc[d]).sum()), int((den & DEL.reindex(L.columns)).sum()), int((hit & DEL.reindex(L.columns)).sum())))
    else:
        # 該月沒有任何快照：用月中位交易日的分母
        dd = L.index[(L.index.to_period('M') == p)]
        if len(dd) == 0: continue
        d = dd[min(7, len(dd) - 1)]; den = L.loc[d]
        mrows.append((p, int(den.sum()), 0, int((den & P300.loc[d]).sum()), 0, int((den & DEL.reindex(L.columns)).sum()), 0))
mm = pd.DataFrame(mrows, columns=['m', 'den', 'hit', 'den300', 'hit300', 'dend', 'hitd']).set_index('m')
mon = {}
for s, (a, b) in glib.SEG.items():
    x = mm[(mm.index >= a.to_period('M')) & (mm.index <= b.to_period('M'))]
    mon[s] = {'cov_full': round(x.hit.sum() / x.den.sum(), 4), 'cov_pool300': round(x.hit300.sum() / x.den300.sum(), 4), 'cov_delisted': round(x.hitd.sum() / max(x.dend.sum(), 1), 4),
              'months_with_snapshot': int((x.hit > 0).sum()), 'months': len(x)}
    mon[s]['pass'] = mon[s]['cov_full'] >= TH and mon[s]['cov_pool300'] >= TH and mon[s]['cov_delisted'] >= TH
mm2 = mm[mm.hit > 0]
mon['from_2014_01'] = {'cov_full': round(mm2.hit.sum() / mm2.den.sum(), 4), 'cov_pool300': round(mm2.hit300.sum() / mm2.den300.sum(), 4), 'cov_delisted': round(mm2.hitd.sum() / max(mm2.dend.sum(), 1), 4), 'months': len(mm2),
                       'first': str(mm2.index.min()), 'last': str(mm2.index.max())}
res['QFIIS']['monthly_variant_descriptive'] = mon
res['QFIIS']['verdict'] = res['QFIIS']['daily']['verdict_raw']
# ---------- MAINT ----------
res['MAINT'] = {'official_per_stock_maintenance_ratio_sources': 0, 'margin_limit_raw_sources': {'2006~2025-09': 0, '2025-09-24 起': 'margin_hist util（使用率，已是衍生比值，非原始限額）'},
                'market_margin_money': 'oid87/PersonalFiance taiwan_margin_total.json（FinMind 衍生，2008-01-02 起，市場總額；非個股）',
                'segments': {s: {'G_PIT': False, 'G_SURV': False, 'G_COV': False, 'cov_full': 0.0} for s in glib.SEG}, 'verdict': '不通過',
                'note': '個股維持率只能用餘額×價格÷融資金額推估（非官方推估，不計覆蓋）；個股融資金額官方不公布'}
json.dump(res, open(HERE / 'data' / 'gate.json', 'w'), ensure_ascii=False, indent=1, default=str)
for k in ['MARGN']:
    for v, r in res[k]['versions'].items():
        print(v, r['n_pass_of_10'], r['verdict_raw'], {s: (r['segments'][s]['cov_full'], r['segments'][s]['cov_pool300'], r['segments'][s]['cov_delisted']) for s in r['segments']}, r['CONSIST'])
print('selected', res['MARGN']['selected'], res['MARGN']['verdict'], res['MARGN']['v1_source_mix'])
q = res['QFIIS']['daily']; print('QFIIS daily', {s: (q['segments'][s]['cov_full'], q['segments'][s]['cov_pool300'], q['segments'][s]['cov_delisted']) for s in q['segments']}, q['verdict_raw'])
print('QFIIS monthly', json.dumps(mon, default=str))
print('MARGN V1 yearly full', mv['V1']['yearly_full']); print('V1 yearly delisted', mv['V1']['yearly_delisted']); print('V1 no-excl', mv['V1']['cov_full_no_newlist_exclusion'])
