#!/usr/bin/env python3
"""前瞻紙上追蹤（SPEC：agents/live_tracking/msci_inclusion/SPEC.json）。
用法：python3 -I -B live_track.py [--check-spec] [--dry-run-review 2026-08]
輸入：agents/live_tracking/msci_inclusion/reviews.tsv（每次 MSCI 公告後人工抄錄：review, ann_tw, eff, cls(STD/SC/TW50/HDETF), code, name, src）。
輸出：events_ledger.csv（第 (i) 層）、strategy_ledger.csv（第 (ii) 層）、status.json（累積檢定與停止規則）。
--dry-run-review：用歷史審核（events_v2）跑一次流程，輸出到 scratchpad，只用來確認程式可執行，不寫入 ledger。"""
import sys, os, json, math, hashlib, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
LT = '/home/user/-/research/tw_stock_selection/agents/live_tracking/msci_inclusion'
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'

def check_spec():
    sp = json.load(open(os.path.join(LT, 'SPEC.json'), encoding='utf-8'))
    body = json.dumps(sp['spec'], ensure_ascii=False, sort_keys=True)
    assert hashlib.sha256(body.encode('utf-8')).hexdigest() == sp['sha256'], 'SPEC.json 被改過'
    return sp

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check-spec', action='store_true'); ap.add_argument('--dry-run-review')
    a = ap.parse_args()
    sp = check_spec()
    if a.check_spec: print('SPEC ok', sp['sha256']); return
    import oa_lib as O
    K, V = O.K, O.V
    IDX, COLS, CI = O.IDX, O.COLS, O.CI
    if a.dry_run_review:
        X = O.inclusion_events(); X = X[X.review == a.dry_run_review].copy()
        X = X[['review', 'ann_tw', 'eff', 'cls', 'code', 'name']].assign(src='events_v2(dry-run)')
        outdir = os.path.join(SCR, 'live_dryrun'); os.makedirs(outdir, exist_ok=True)
    else:
        f = os.path.join(LT, 'reviews.tsv')
        X = pd.read_csv(f, sep='\t', dtype={'code': str}, parse_dates=['ann_tw', 'eff'], comment='#')
        X = X[pd.to_datetime(X.eff) >= pd.Timestamp(sp['spec']['start_review_eff_on_or_after'])].copy()
        outdir = LT
    rows = []
    for r in X.itertuples():
        rec = dict(review=r.review, cls=r.cls, code=r.code, name=r.name, ann_tw=str(pd.Timestamp(r.ann_tw).date()), eff=str(pd.Timestamp(r.eff).date()), src=r.src)
        if r.code not in CI or pd.Timestamp(r.eff) not in IDX:
            rec['status'] = 'not_in_panel_or_E_not_reached'; rows.append(rec); continue
        j = CI[r.code]; i = IDX.get_loc(pd.Timestamp(r.eff))
        v60 = O.V60[i - 1, j]; px = K.Cn[i - 1, j]
        rec['liquid'] = bool(np.isfinite(v60) and v60 >= 1e7 and np.isfinite(px) and px >= 10)
        for h in (-10, 0, 5, 20):
            rec[f'close_E{h:+d}'] = float(K.Cn[i + h, j]) if 0 <= i + h < len(IDX) and np.isfinite(K.Cn[i + h, j]) else None
        rec['car_E-10_E-1'] = O.ar(j, i - 10, i - 1)
        rec['ar_E'] = O.ar(j, i, i)
        rec['car_E+1_E+5'] = O.ar(j, i + 1, i + 5) if i + 5 < len(IDX) else None
        rec['car_E+1_E+20'] = O.ar(j, i + 1, i + 20) if i + 20 < len(IDX) else None
        rec['status'] = 'complete' if i + 20 < len(IDX) else ('E+5_done' if i + 5 < len(IDX) else 'pending')
        rows.append(rec)
    EV = pd.DataFrame(rows)
    EV.to_csv(os.path.join(outdir, 'events_ledger.csv'), index=False)
    # (ii) 主策略受影響事件：排除金融 n10（與 n30），持有期內遇到 STD 生效日
    S = V.V2('gapfix', O.YL, O.Y5f, latest=True, exfin=True)
    srows = []
    for n in (10, 30):
        pl = S.plan(n, 'official'); ex = [p[0] for p in pl]
        for k, (e, pk, pool, d) in enumerate(pl):
            e1 = ex[k + 1] if k + 1 < len(ex) else None
            held = {COLS[q] for q in pk}
            for r in X[(X.cls == 'STD') & X.code.isin(held)].itertuples():
                E = pd.Timestamp(r.eff)
                if E not in IDX: continue
                iE = IDX.get_loc(E)
                if not (e <= iE and (e1 is None or iE <= e1 - 1)): continue
                j = CI[r.code]
                if e1 is None or e1 >= len(IDX):
                    srows.append(dict(n=n, review=r.review, code=r.code, sig=str(d.date()), E=str(E.date()), status='pending_next_rebalance')); continue
                A = K.MYADJ.to_numpy()
                Ri = (K.On[e1, j] * A[e1, j] / K.Cn[e1, j]) / A[iE, j] - 1 if K.TRADED[e1, j] else A[e1 - 1, j] / A[iE, j] - 1
                r50 = O.E50[e1 - 1] / O.E50[iE] - 1
                w = 1.0 / n
                srows.append(dict(n=n, review=r.review, code=r.code, sig=str(d.date()), E=str(E.date()), next_exec=str(IDX[e1].date()),
                                  locked_limit_down_E=O.limit_dn_locked(iE, j), R_stock_E_to_next=Ri, R_0050=r50,
                                  paper_diff_B2_minus_orig=w * (r50 - Ri - 0.0069), status='complete'))
    SL = pd.DataFrame(srows); SL.to_csv(os.path.join(outdir, 'strategy_ledger.csv'), index=False)
    # 累積檢定
    st = {'n_reviews_complete': int(EV[EV.status.isin(['complete', 'E+5_done'])].review.nunique()) if len(EV) else 0}
    g = EV[(EV.cls == 'STD') & EV.get('liquid', pd.Series(dtype=bool)).fillna(False) & EV['car_E+1_E+5'].notna()] if len(EV) and 'car_E+1_E+5' in EV else pd.DataFrame()
    if len(g):
        m = g.groupby('review')['car_E+1_E+5'].mean()
        st['STD_E+1_E+5_mean_clustered%'] = round(float(m.mean()) * 100, 2)
        st['t_clustered'] = round(float(m.mean() / (m.std(ddof=1) / math.sqrt(len(m)))), 2) if len(m) > 2 else None
        st['n_STD_events'] = int(len(g))
    if len(SL) and 'paper_diff_B2_minus_orig' in SL:
        st['strategy_cum_diff_n10%'] = round(float(SL[(SL.n == 10)].paper_diff_B2_minus_orig.sum()) * 100, 3)
    rule = sp['spec']['decision']
    if st['n_reviews_complete'] >= rule['min_reviews'] and 'STD_E+1_E+5_mean_clustered%' in st:
        if st['STD_E+1_E+5_mean_clustered%'] > rule['close_if_event_mean_gt%']: st['decision'] = '關閉此題'
        elif st['STD_E+1_E+5_mean_clustered%'] <= rule['event_mean_le%'] and (st.get('t_clustered') or 0) <= rule['event_t_le'] and st.get('strategy_cum_diff_n10%', -1) >= 0:
            st['decision'] = '建議寫進 SOP（仍須總監核准）'
        else: st['decision'] = '繼續追蹤'
    else:
        st['decision'] = f"累積中（需 ≥ {rule['min_reviews']} 次審核）"
    json.dump(st, open(os.path.join(outdir, 'status.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False))

if __name__ == '__main__':
    main()
