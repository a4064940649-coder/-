#!/usr/bin/env python3
"""S68 第 0 步：資料盤點（不算任何策略報酬）。python3 -I -B inventory.py → data_inventory.json
盤點：上櫃日 K（收、開、成交金額、漲跌價差）、市場別旗標、下市櫃、月營收（含 2018-10 補洞）、除權息來源（TWT49U 只上市；本地 exdiv_5y 的上櫃部分）、
上櫃除權息日在官方日 K 的樣子（漲跌價差缺值）與 exdiv_5y 的對照、股池大小。"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s68lib as L
from s68lib import cslib, stlib, s4lib, s17lib

H = Path(__file__).resolve().parent
t0 = time.time()
out = dict(cache_sha256=L.setup_cache())
z = np.load(L.CACHE / 'panel_cs.npz', allow_pickle=False)
idx = pd.DatetimeIndex(z['IDX']); keep = idx <= cslib.END
IDX = idx[keep]; COLS = z['COLS'].astype(str)
C, O, AMT, CHG, MK, VOL = (z[k][keep] for k in ('C', 'O', 'AMT', 'CHG', 'MK', 'VOL'))
T, N = C.shape
has = np.isfinite(C) & (C > 0)
yrs = np.array(IDX.year)
mk_name = {1: '上市', 2: '上櫃'}
pan = dict(T=T, N=N, first=str(IDX[0].date()), last=str(IDX[-1].date()),
           market_flag='P.MK（T×N int8，逐日）：1＝上市（twse）、2＝上櫃（tpex）、0＝當日不在官方日 K；stlib.universe(P, "twse")＝P.liq & (MK==1)，其他值＝P.liq & (MK>0)；s20lib.load() 本身不分市場，面板上市＋上櫃都載入',
           source='raw/jb7k6295r7-bot_tw-stock-data data/universe/daily/YYYY-MM-DD.csv（官方日 K，上市＋上櫃；四碼非 0 開頭）；本題用官方快取 panel_cs.npz（sha256 驗證）')
by = {}
for m in (1, 2):
    k = MK == m
    first = np.array([IDX[np.flatnonzero(k[:, j])[0]] if k[:, j].any() else pd.NaT for j in range(N)])
    last_tr = np.array([np.flatnonzero(has[:, j] & k[:, j]).max() if (has[:, j] & k[:, j]).any() else -1 for j in range(N)])
    ever = k.any(axis=0)
    lastmk = np.array([np.flatnonzero(MK[:, j] > 0).max() if (MK[:, j] > 0).any() else -1 for j in range(N)])
    endm = np.array([MK[lastmk[j], j] if lastmk[j] >= 0 else 0 for j in range(N)])
    delisted = ever & (endm == m) & (lastmk < T - 5)                 # 最後一根日 K 早於資料結束前 5 個交易日，且當時在這個市場
    d = dict(n_codes_ever=int(ever.sum()), first_date=str(pd.Timestamp(min(x for x in first if x is not pd.NaT)).date()),
             last_date=str(IDX[np.flatnonzero(k.any(axis=1)).max()].date()),
             n_delisted=int(delisted.sum()), delisted_by_year={str(y): int((delisted & (yrs[np.maximum(lastmk, 0)] == y)).sum()) for y in sorted(set(yrs))},
             codes_per_year={str(y): int(k[yrs == y].any(axis=0).sum()) for y in sorted(set(yrs))},
             cells_listed=int(k.sum()), cells_traded=int((k & has).sum()),
             nan_frac_when_traded={n: float((~np.isfinite(a))[k & has].mean()) for n, a in (('close', C), ('open', O), ('amount', AMT), ('change', CHG), ('volume', VOL))},
             change_nan_cells=int((k & has & np.isnan(CHG)).sum()), change_zero_cells=int((k & has & (CHG == 0)).sum()))
    with np.errstate(invalid='ignore', divide='ignore'):
        q = AMT / (VOL * C)
    d['amount_over_vol_x_close_median'] = float(np.nanmedian(q[k & has]))
    by[mk_name[m]] = d
pan['by_market'] = by
tr = [(str(COLS[j]), [int(v) for v in pd.unique(MK[:, j][MK[:, j] > 0])]) for j in range(N) if len(set(MK[:, j][MK[:, j] > 0])) > 1]
pan['market_switch'] = dict(n_otc_to_twse=sum(1 for c, s in tr if s[0] == 2), n_twse_to_otc=sum(1 for c, s in tr if s[0] == 1))
out['panel'] = pan

# ---- 除權息來源
tw = cslib.load_twt49u()
ex = pd.read_csv(L.EXDIV_5Y, dtype={'code': str}); ex['date'] = pd.to_datetime(ex['date'])
ex4 = ex[ex['code'].str.fullmatch(r'[1-9]\d{3}')]
ev = L.load_exdiv_tpex()
out['exright_sources'] = dict(
    twt49u=dict(path=str(cslib.TWT49U_DIR), market='只有上市（README 明載；檔內無市場欄，代號全為上市／ETF）', first=str(tw['date'].min().date()), last=str(tw['date'].max().date()), n_4digit=int(len(tw)),
                n_codes_that_are_otc_on_event_day=None),
    exdiv_5y=dict(path=str(L.EXDIV_5Y), first=str(ex['date'].min().date()), last=str(ex['date'].max().date()),
                  rows_by_market={k: int(v) for k, v in ex4['market'].value_counts().items()},
                  tpex_by_year={str(y): int(v) for y, v in ev.groupby(ev['date'].dt.year).size().items()},
                  tpex_kind={str(k): int(v) for k, v in ev['kind'].value_counts().items()},
                  note='tw-stock-screener 公開 repo 的 extras；含上櫃（TPEX）除權息參考價，但只有 2021-10-07 起；2015-01～2021-09 的上櫃除權息參考價本地沒有'),
    conclusion='上櫃除權息參考價：2021-10-07～2026-10-07 有（exdiv_5y）；2015-01-05～2021-10-06 沒有任何本地來源。')
ci = {c: j for j, c in enumerate(COLS)}; di = {d: i for i, d in enumerate(IDX)}
tw_in = tw[tw['code'].isin(ci) & tw['date'].isin(di)]
out['exright_sources']['twt49u']['n_codes_that_are_otc_on_event_day'] = int((MK[tw_in['date'].map(di).to_numpy(), tw_in['code'].map(ci).to_numpy()] == 2).sum())

# ---- 上櫃除權息日在官方日 K 的樣子
zc = L.otc_cells(IDX, COLS, C, CHG, MK)
a, b, prevC, REFX = zc['a'], zc['b'], zc['prevC'], zc['REFX']
otc = MK == 2
d0 = int(np.searchsorted(IDX.values, np.datetime64('2021-10-07')))
d1 = int(np.searchsorted(IDX.values, np.datetime64('2026-10-07'), side='right'))
win = np.zeros((T, N), bool); win[d0:d1] = True
hasref = np.isfinite(REFX) & otc
with np.errstate(invalid='ignore', divide='ignore'):
    rr = C / REFX; rp = C / prevC
    inb = (rr >= 0.899) & (rr <= 1.101)
    ex_ret = (rr - 1)[a & hasref & inb]
    refprev = (REFX / prevC - 1)[a & hasref & inb]
o = dict(
    a_change_nan_prev_traded=int(a.sum()), a_by_year={str(y): int(a[yrs == y].sum()) for y in sorted(set(yrs))},
    a_by_month={str(m): int(a[np.array(IDX.month) == m].sum()) for m in range(1, 13)},
    a_mean_close_over_prev_pct=float(np.mean(rp[a] - 1) * 100), a_frac_close_below_prev=float((rp[a] < 1).mean()),
    a_abs_move_gt_10p5pct=int((np.abs(rp[a] - 1) > 0.105).sum()),
    change_nan_other=dict(prev_untraded_or_first_bar=int((otc & has & np.isnan(CHG) & ~a).sum())),
    b_change_zero_prev_untraded_le_10p5=int(b.sum()),
    change_zero_price_changed_prev_traded=int((otc & has & (CHG == 0) & np.isfinite(prevC) & (np.abs(C - prevC) > 1e-9) & np.vstack([np.zeros((1, N), bool), has[:-1]])).sum()),
    overlap_with_exdiv_5y_window=dict(
        window='2021-10-07～2026-10-07',
        exdiv_tpex_events_in_panel=int((hasref & win).sum()), exdiv_events_traded=int((hasref & has).sum()),
        a_cells_in_window=int((a & win).sum()), a_and_exdiv=int((a & hasref).sum()),
        a_without_exdiv=int((a & win & ~hasref).sum()), exdiv_without_a=int((hasref & has & ~a).sum()),
        exdiv_without_a_change_numeric=int((hasref & has & np.isfinite(CHG)).sum()),
        a_and_exdiv_close_within_10p1pct_of_ref=int((a & hasref & inb).sum()), a_and_exdiv_outside=int((a & hasref & ~inb).sum()),
        prev_close_matches_exdiv=None,
        exday_close_over_ref_mean_pct=float(ex_ret.mean() * 100), exday_close_over_ref_median_pct=float(np.median(ex_ret) * 100),
        ref_over_prev_mean_pct=float(refprev.mean() * 100)),
    reading='上櫃官方日 K 的漲跌價差在除權息日是「缺值」（櫃買日報該欄是文字「除息／除權」），不是 0。共用引擎遇缺值改用收盤÷前收−1＝把股利當成下跌（偏悲觀）；'
            'EXRIGHT_FIX 只修上市（TWT49U）。S16 的上櫃結果就是在這個偏悲觀的口徑下算的。')
e2 = ev[ev['code'].isin(ci) & ev['date'].isin(di)]
pc = prevC[e2['date'].map(di).to_numpy(), e2['code'].map(ci).to_numpy()]
pe = pd.to_numeric(e2['prev_close'], errors='coerce').to_numpy()
o['overlap_with_exdiv_5y_window']['prev_close_matches_exdiv'] = dict(n=int(np.isfinite(pc).sum()), match=int((np.abs(pc - pe) < 0.011).sum()))
out['otc_exright_in_daily_k'] = o

# ---- 月營收
r = pd.read_csv(L.REV_ORIG, dtype={'code': str}, parse_dates=['month'])
cnt = r.groupby(['market', r['month'].dt.to_period('M')])['code'].nunique().unstack(0)
n_add, psha = L.build_patched_rev()
ro = r[r['market'] == 'otc']
otc_codes = set(COLS[(MK == 2).any(axis=0)])
lastk = {COLS[j]: IDX[np.flatnonzero(MK[:, j] > 0).max()] for j in range(N) if (MK[:, j] > 0).any()}
deli = [c for c in otc_codes if lastk[c] < IDX[-6] and MK[np.flatnonzero(MK[:, ci[c]] > 0).max(), ci[c]] == 2]
lastrev = r.groupby('code')['month'].max()
ok3 = sum(1 for c in deli if c in lastrev.index and (lastk[c].to_period('M') - lastrev[c].to_period('M')).n <= 3)
out['revenue'] = dict(path=L.REV_ORIG, markets={str(k): int(v) for k, v in r['market'].value_counts().items()},
                      otc_first=str(ro['month'].min().date()), otc_last=str(ro['month'].max().date()),
                      otc_n_per_year_median={str(y): float(cnt['otc'][cnt.index.year == y].median()) for y in range(2013, 2027)},
                      otc_hole_months=[(str(m), int(v)) for m, v in cnt['otc'].items() if m.year >= 2013 and v < 0.7 * cnt['otc'].rolling(13, center=True, min_periods=3).median()[m]],
                      patch=dict(file=str(L.PATCH_CSV), sha256=L.PATCH_SHA, rows_added=n_add, patched_sha256=psha, note='S16 用 MOPS 官方彙總頁補 2018-10 上櫃 616 家（已獨立複製驗證）'),
                      otc_delisted_codes=len(deli), otc_delisted_with_revenue_to_within_3m=int(ok3),
                      note='2013 起營收檔只剩現存公司 → 下市櫃公司選不到（上市同樣限制）；偏誤方向＝高估（存活者），S16 複製員量過約 −0.15 個百分點／年；KY 公司不在營收檔。')
out['benchmark'] = dict(path=cslib.E50_CSV, note='0050 含息還原，同現行')
out['costs'] = '上櫃手續費、證交稅 0.3%、跳動點、漲跌幅 ±10%、零股上路日同上市；引擎不分市場（S16 已查）。上櫃實際買賣價差可能較大 → 成本偏樂觀。'
out['verdict'] = dict(sufficient=True,
                      gaps=['2015-01～2021-10-06 上櫃除權息參考價本地沒有 → 近似：除權息日含息報酬記 0（上市稽核：除權息日收盤對參考價平均 +0.36%；本檔上櫃 2021-10 起實測見 exday_close_over_ref_mean_pct）→ 偏誤方向：略偏悲觀（少算除權息日當天的填息）',
                            '下市櫃公司 2013 起沒有營收 → 選不到，偏樂觀（小）', '上櫃成本模型同上市 → 偏樂觀', '創新板旗標只在上市；上櫃無對應板別'])
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(H / 'data_inventory.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
print(json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))[:9000])
