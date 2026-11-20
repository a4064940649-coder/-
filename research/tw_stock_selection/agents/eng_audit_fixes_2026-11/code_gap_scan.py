#!/usr/bin/env python3
"""共用引擎兩個資料問題的影響範圍：(1) 代號重用（下市後同代號給另一家公司）、(2) 面板中間長期沒有價格。
引擎的下市判斷是「這一天不在面板、而且之後再也沒出現」（s10lib.py 第 314 行）；代號之後又出現（重用或資料缺一段）就不算下市，
部位會停在最後市值、不能賣，直到那個代號再出現。
本程式：掃面板找出所有「上市列中斷 ≥ 60 個交易日後又出現」的代號，再對各研究的交易明細找出持有期間跨到中斷的交易。只讀。
python3 -I -B code_gap_scan.py → gaps/panel_gaps_*.csv、gaps/affected_trades.csv、gaps/gap_scan.json"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent; OUT = HERE / 'gaps'
PANELS = {'連續面板_2007_2026（S75～S83、R75 用）': '/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s75/cache_cont/panel_cs.npz',
          '正式面板_2015_2026（正式引擎用）': '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/offfix/cache/panel_cs.npz'}
KNOWN11 = ['6176', '8033', '6205', '6283', '8021', '9955', '5533', '2059', '4108', '6226', '8103']
MIN_GAP = 60
R = {}; G = {}
for nm, p in PANELS.items():
    if not Path(p).exists():
        R[nm] = '面板快取不在，沒掃'; continue
    z = np.load(p, allow_pickle=False)
    IDX = pd.DatetimeIndex(z['IDX']); COLS = z['COLS'].astype(str); MK = z['MK']; C = z['C']
    rows = []
    for j, c in enumerate(COLS):
        on = np.flatnonzero(MK[:, j] > 0)
        if len(on) < 2:
            continue
        brk = np.flatnonzero(np.diff(on) > MIN_GAP)
        for b in brk:
            a, e = on[b], on[b + 1]
            tr = np.flatnonzero(np.isfinite(C[:a + 1, j]) & (C[:a + 1, j] > 0)); tr2 = np.flatnonzero(np.isfinite(C[e:, j]) & (C[e:, j] > 0)) + e
            ca = float(C[tr[-1], j]) if len(tr) else np.nan; ce = float(C[tr2[0], j]) if len(tr2) else np.nan
            rows.append(dict(code=c, last_day_before=str(IDX[a].date()), first_day_after=str(IDX[e].date()), gap_trading_days=int(e - a - 1),
                             market_before=int(MK[a, j]), market_after=int(MK[e, j]), close_before=ca, close_after=ce,
                             price_ratio=round(ce / ca, 3) if ca == ca and ce == ce and ca > 0 else None,
                             known=('2432 代號重用' if c == '2432' else ('S77 列的 11 檔' if c in KNOWN11 else '新發現'))))
    g = pd.DataFrame(rows); fn = 'panel_gaps_' + ('cont' if '連續' in nm else 'official') + '.csv'; g.to_csv(OUT / fn, index=False)
    G[nm] = g
    R[nm] = dict(range=[str(IDX[0].date()), str(IDX[-1].date())], n_codes=int(len(COLS)), n_gaps=int(len(g)), n_codes_with_gap=int(g['code'].nunique()) if len(g) else 0,
                 known11_found=sorted(set(g['code']) & set(KNOWN11)) if len(g) else [], has_2432=bool(len(g) and (g['code'] == '2432').any()),
                 new_codes=sorted(set(g.loc[g['known'] == '新發現', 'code'])) if len(g) else [], file=fn)
    del z, MK, C
# ---- 各研究交易明細
TR = {'S75 連續帳戶（trades.csv.gz）': ('s75_continuous_2008_2026/trades.csv.gz', 'version'),
      'S76 前後段診斷（trades_tagged）': ('s76_early_vs_late_diagnosis/trades_tagged.csv.gz', None),
      'S77 前段訂規則（trades.csv.gz）': ('s77_rule_from_early_period/trades.csv.gz', 'rule'),
      'S79 凍結格（trades_frozen）': ('s79_revenue_plus_other_signals/trades_frozen.csv.gz', None),
      'S81 重點格（trades_key）': ('s81_breadth_fill_hold/trades_key.csv.gz', None),
      'S82 大贏家（trades_features，進行中）': ('s82_big_winner_dependence/trades_features.csv.gz', None),
      '稽核逐筆重算（trades_recalc）': ('audit_data_2008_2026/trades_recalc.csv.gz', None),
      '正式引擎 E0｜W10_70 標準路徑（本次倒出）': ('eng_audit_fixes_2026-11/reduce/trades_E0.csv.gz', None)}
aff = []; R['studies'] = {}
for nm, (f, vcol) in TR.items():
    p = AG / f
    if not p.exists():
        R['studies'][nm] = '檔案不在'; continue
    t = pd.read_csv(p, dtype={'code': str}, low_memory=False)
    dc = {k: next((c for c in t.columns if c in cand), None) for k, cand in (('ent', ('ent_date', 'entry_date', 'buy_date', 'ent')), ('exit', ('exit_date', 'sell_date', 'exit')))}
    if 'code' not in t.columns or not dc['ent'] or not dc['exit']:
        R['studies'][nm] = f'欄位認不得：{list(t.columns)[:14]}'; continue
    t['_e'] = pd.to_datetime(t[dc['ent']]); t['_x'] = pd.to_datetime(t[dc['exit']])
    g = G['正式面板_2015_2026（正式引擎用）' if nm.startswith('正式') else '連續面板_2007_2026（S75～S83、R75 用）']
    grp = [c for c in t.columns if c in ('version', 'rule', 'cell', 'variant', 'tag', 'key', 'label', 'sid', 'name')]
    n_aff = 0; vers = sorted(t[vcol].astype(str).unique()) if vcol else None
    for r in g.itertuples():
        a, e = pd.Timestamp(r.last_day_before), pd.Timestamp(r.first_day_after)
        m = t[(t['code'] == r.code) & (t['_e'] <= e) & (t['_x'] > a)]
        for x in m.itertuples():
            n_aff += 1
            aff.append(dict(study=nm, group='|'.join(str(getattr(x, c)) for c in grp), code=r.code, issue=r.known, gap=f'{r.last_day_before}→{r.first_day_after}',
                            ent_date=str(x._e.date()), exit_date=str(x._x.date()), why=getattr(x, 'why', ''), cost=round(float(getattr(x, 'cost', np.nan))), proceeds=round(float(getattr(x, 'proceeds', np.nan))),
                            pnl=round(float(getattr(x, 'pnl', np.nan))), ret_pct=round(float(getattr(x, 'ret', np.nan)) * 100, 2), hold_days=int(getattr(x, 'hold_days', -1))))
    # 另外：2432 在 2023-05 之後（新公司）被買過嗎
    n2432 = t[(t['code'] == '2432')]; nk11 = t[t['code'].isin(KNOWN11)]
    R['studies'][nm] = dict(file=f, n_trades=int(len(t)), period=[str(t['_e'].min().date()), str(t['_x'].max().date())], group_cols=grp, versions=vers, n_trades_crossing_gap=n_aff,
                            n_trades_2432_any=int(len(n2432)), trades_2432=[(str(a.date()), str(b.date())) for a, b in zip(n2432['_e'], n2432['_x'])][:10],
                            n_trades_known11_any=int(len(nk11)), n_trades_known11_before_2010=int((nk11['_e'] < '2010-01-04').sum()))
A = pd.DataFrame(aff); A.to_csv(OUT / 'affected_trades.csv', index=False)
R['n_affected_trades_total'] = int(len(A))
(OUT / 'gap_scan.json').write_text(json.dumps(R, ensure_ascii=False, indent=1))
print(json.dumps(R, ensure_ascii=False, indent=1))
for nm, g in G.items():
    print(nm); print(g.to_string())
if len(A):
    print(A.to_string())
