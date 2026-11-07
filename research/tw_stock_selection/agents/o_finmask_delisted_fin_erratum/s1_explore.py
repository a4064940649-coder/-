#!/usr/bin/env python3
"""第 1 步探索（只看分類與 C0 現行名次，不算任何 F 遮罩下的名單異動）：
A 分類器 v1 盤點；B 池內已下市金融股清單（下市日、原因、池內月份、C0 最佳名次）；C 28xx 涵蓋核對；
D 管線有效性（C0 原樣重跑 0 差異、U_opt 原樣重跑與 ER2 逐筆相同、5 日獨立重算）；E 處置資料與營收期限事實。
輸出 explore.json。用法：python3 -I -B s1_explore.py"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freg as R
R.check_reg('PREREG_DESIGN')
import fclass as F
import flib as L
import numpy as np, pandas as pd
E, V, K = L.E, L.V, L.K
T0 = time.time()
KW_V1 = ['銀行', '商銀', '金控', '金融', '證券', '證', '票券', '保險', '人壽', '產險', '期貨', '投信']
CL = F.Classifier(KW_V1)
out = dict(utc=R.now_utc(), note='探索：只看分類清單、C0 現行名單與名次；沒有計算任何 F 遮罩下的名單或異動')

# ---------- D 管線有效性（先做） ----------
L0 = json.loads((R.AG / 'o_g8_protocol_fin_erratum/ERRATUM.json').read_text(encoding='utf-8'))['lists_C0']
ER2 = json.loads((R.AG / 'o_rev_trunc_2009_erratum/ERRATUM2.json').read_text(encoding='utf-8'))
R0 = L.run(lambda d: L.C0MASK, 'base')
days = sorted(R0); D1 = [d for d in days if d < '2013-01-01']
diff0 = sum(len(set(R0[d]['picks']) ^ set(L0[d])) for d in days) + abs(len(days) - len(L0))
order_same = all(R0[d]['picks'] == L0[d] for d in days)
RU = L.run(lambda d: L.C0MASK, 'opt')
slots_u = sorted([list(x) for x in L.inserted(RU)])
er2_slots = sorted([list(x) for x in ER2['U_opt_slot_list']])
out['D_pipeline'] = dict(n_days=len(days), n_days_2006_2012=len(D1), C0_rerun_sym_diff=diff0, C0_rerun_order_identical=order_same,
                         U_opt_rerun_n=len(slots_u), U_opt_rerun_n_2006_2012=sum(1 for d, c in slots_u if d < '2013'),
                         U_opt_rerun_equals_ER2=(slots_u == er2_slots), ER2_n=len(er2_slots))
assert diff0 == 0 and len(days) == 93 and len(D1) == 81, 'C0 原樣重跑不一致，管線不可用'
assert slots_u == er2_slots, 'U_opt 原樣重跑與 ER2 不一致'
print('pipeline OK', round(time.time() - T0), flush=True)

# 5 日獨立重算（C0 遮罩；不經 V2.cross_section／order／plan）
S52, V60 = V.rolling_signals('gapfix')
SAMPLE = ['2006-04-11', '2008-04-11', '2009-11-11', '2011-07-11', '2013-12-11']
def indep(d, mask, held):
    D = pd.Timestamp(d); t = K.IDX.get_loc(D); e = t + 1
    top = L.top300(D)
    y = K.yoy3_for_month(E.YL, pd.Period(D, 'M')); y = np.where(mask, np.nan, y)
    s = S52[t]
    el = top & np.isfinite(y) & np.isfinite(s) & ~K.disposal_mask(D) & np.isfinite(K.On[e]) & K.TRADED[e]
    idx = np.where(el)[0]
    r1 = pd.Series(s[idx]).rank(pct=True).to_numpy(); r2 = pd.Series(y[idx]).rank(pct=True).to_numpy()
    sc = np.round(0.5 * r1 + 0.5 * r2, 10)
    srt = sorted(range(len(idx)), key=lambda i: (-sc[i], -y[idx[i]], idx[i]))
    gu = K.gap_up_open(e); pk = []
    for i in srt:
        j = idx[i]
        if gu[j] and L.COLS[j] not in held: continue
        pk.append(L.COLS[j])
        if len(pk) == 10: break
    return pk
samp = {}
for d in SAMPLE:
    i = days.index(d); held = set(R0[days[i - 1]]['picks']) if i > 0 else set()
    pk = indep(d, L.C0MASK, held); samp[d] = dict(indep=pk, pipeline=R0[d]['picks'], same=(pk == R0[d]['picks']))
out['D_sample5_indep_C0'] = samp
assert all(v['same'] for v in samp.values()), '獨立重算不一致'
print('sample5 OK', flush=True)

# ---------- A 分類器 v1 盤点 ----------
POOL = {d: L.top300(pd.Timestamp(d)) for d in days}
st_cnt = {}; unk = {}; f1_vs_c0 = {}
for d in days:
    for j in np.where(POOL[d])[0]:
        c = L.COLS[j]; s, src = CL.f1(c, d)
        st_cnt[s] = st_cnt.get(s, 0) + 1
        v = CL.f1_value(s, L.C0MASK[j])
        if s == 'UNKNOWN': unk.setdefault(c, dict(name='|'.join(sorted(F.NAMES.get(c, []))), src=set(), n=0, C0=bool(L.C0MASK[j]), first=d, last=d)); unk[c]['n'] += 1; unk[c]['src'].add(src); unk[c]['last'] = d
        if v != bool(L.C0MASK[j]):
            f1_vs_c0.setdefault(c, dict(name='|'.join(sorted(F.NAMES.get(c, []))), C0=bool(L.C0MASK[j]), F1=v, states=set(), n_pool_months=0, first=d, last=d))
            f1_vs_c0[c]['n_pool_months'] += 1; f1_vs_c0[c]['states'].add(s); f1_vs_c0[c]['last'] = d
f2set = {c for c in L.COLS if CL.f2(c)}
c0set = {c for j, c in enumerate(L.COLS) if L.C0MASK[j]}
pool_codes = sorted({L.COLS[j] for d in days for j in np.where(POOL[d])[0]})
out['A_classifier_v1'] = dict(
    kw=KW_V1, f1_state_counts_pool_stockmonths=st_cnt,
    unknown_in_pool={c: dict(v, src=sorted(v['src'])) for c, v in unk.items()},
    f1_differs_from_c0_in_pool={c: dict(v, states=sorted(v['states'])) for c, v in f1_vs_c0.items()},
    f2_n=len(f2set), c0_n=len(c0set), f2_minus_c0=sorted(f2set - c0set), c0_minus_f2=sorted(c0set - f2set),
    f2_minus_c0_in_pool=sorted((f2set - c0set) & set(pool_codes)),
    keyword_hits_without_prefix=sorted(c for c in L.COLS if c[:2] not in ('28', '58', '60') and CL.kw_hit(c)),
    snapshots=[dict(rel=s['rel'], eff=str(s['date'].date()), n=len(s['fin']), sha256=R.sha(F.O13 / s['rel'])) for s in F.SNAPS],
    later_sources=[dict(rel=s['rel'], eff=str(s['date'].date()), n=len(s['fin'])) for s in F.LATER])
print('A done', round(time.time() - T0), flush=True)

# ---------- B 池內已下市金融股（F1 FIN／FIN_Q、F2 FIN、或 F1 UNKNOWN） ----------
REASON = {  # 研究員記憶，未查證（證交所／公開資訊觀測站經代理 403，無法補抓）
    '5854': '股份轉換：合作金庫銀行以股份轉換成立合庫金控（5880，2011-12-01 上市），合庫銀下市',
    '6004': '合併：元京證券與元大證券合併（2007-04，元京證消滅），屬元大金控集團整併',
    '6012': '股份轉換：金鼎證券以股份轉換成為開發金控子公司（2011-05），之後併入凱基證券',
}
rows = []
for c in pool_codes:
    j = L.CI[c]
    if not F.is_delisted(c): continue
    f1s = sorted({CL.f1(c, d)[0] for d in days if POOL[d][j]})
    f2 = CL.f2(c)
    if not (f2 or any(s in ('FIN', 'FIN_Q', 'UNKNOWN') for s in f1s)): continue
    lm = F.LM.get(c, {})
    pm = [d for d in days if POOL[d][j]]
    elm = [d for d in pm if R0[d]['el'][j]]
    rk = [R0[d]['rank'].get(c) for d in elm]
    picks = [d for d in days if c in R0[d]['picks']]
    w = np.where(K.TRADED[:, j])[0]
    rows.append(dict(code=c, name='|'.join(sorted(F.NAMES.get(c, []))), C0=bool(L.C0MASK[j]), F1_states_in_pool=f1s, F2=f2,
                     listed=lm.get('listed_date', ''), delist_lm=lm.get('delist_date', ''), delist_official=F.DEL.get(c, []),
                     last_trade=str(K.IDX[w[-1]].date()) if len(w) else '', reason=REASON.get(c, '28xx（C0 已遮）' if c[:2] == '28' else ''),
                     reason_verified=False, n_pool_months=len(pm), pool_first=pm[0] if pm else '', pool_last=pm[-1] if pm else '',
                     n_el_months=len(elm), best_rank_C0=min(rk) if rk else None, C0_pick_dates=picks,
                     pool_by_year={y: sum(1 for d in pm if d[:4] == y) for y in sorted({d[:4] for d in pm})}))
out['B_delisted_fin_in_pool'] = rows
out['B_missed_by_C0'] = [r['code'] for r in rows if not r['C0']]
out['B_reason_note'] = '下市原因為研究員記憶，未查證；listing_master.csv 沒有下市原因欄，證交所／公開資訊觀測站／archive.org 經代理一律 403'
# ---------- C 28xx 涵蓋 ----------
d28 = [r for r in rows if r['code'][:2] == '28']
out['C_28xx_delisted_in_pool'] = dict(codes=[r['code'] for r in d28], all_masked_by_C0=all(r['C0'] for r in d28),
                                      n_pool_months=sum(r['n_pool_months'] for r in d28), check_2847=next((r for r in rows if r['code'] == '2847'), None))
# ---------- E 事實 ----------
dsp = pd.read_csv(K.X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
off = {}
for f in ('official_trades_twse.csv', 'official_trades_tpex.csv'):
    p = R.AG / 'disposal_reconcile' / f
    if p.exists():
        T = pd.read_csv(p, dtype=str)
        dc = [c for c in T.columns if 'start' in c.lower() or 'date' in c.lower() or '日' in c]
        off[f] = dict(sha256=R.sha(p), cols=list(T.columns), min={c: str(pd.to_datetime(T[c], errors='coerce').min()) for c in dc})
dl = {d: V.deadline_case(pd.Timestamp(d)) for d in days}
out['E_facts'] = dict(disposal_5y=dict(path=str(K.X / 'disposal_5y.csv.gz'), sha256=R.sha(K.X / 'disposal_5y.csv.gz'), min_start=str(dsp['start'].min().date()),
                                       min_date=str(dsp['date'].min().date()), n=len(dsp),
                                       n_mask_hits_2006_2013=int(sum(K.disposal_mask(pd.Timestamp(d)).sum() for d in days))),
                      disposal_official=off,
                      deadline_case_counts=pd.Series(dl).value_counts().to_dict(), DS_days=[d for d in days if dl[d] == 'D=S'])
out['runtime_s'] = time.time() - T0
(HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: out[k] for k in ('D_pipeline', 'B_missed_by_C0', 'C_28xx_delisted_in_pool')}, ensure_ascii=False, default=str, indent=0)[:3000])
print(json.dumps(out['A_classifier_v1'], ensure_ascii=False, default=str)[:4000])
for r in rows: print(r)
print(out['E_facts']['disposal_5y'], out['E_facts']['deadline_case_counts'], len(out['E_facts']['DS_days']))
