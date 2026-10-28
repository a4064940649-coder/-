#!/usr/bin/env python3
"""步驟 1：上市身分母體（只看資料，不看報酬）→ listing_master.csv、data/listed_twse.parquet（逐日上市矩陣）、data/listing_review.json。

市場別證據（全部取自 raw/ 或 V6 面板的 src 欄）：
  TW（上市）：V6 面板 src∈{1,2,4}（證交所官方行情）、src 3（Yahoo「.TW」＝上市代號，弱）、leochen（證交所 STOCK_DAY）、bacardi market=TWSE、
              Tasi 淨值「市」檔（逐日；名單不完整）、Tasi 營收月檔「全部國內上市公司合計」之前（MOPS 上市彙總，月；證據日＝次月 10 日）、
              ISIN 封存快照（GitHub chenFlyingkite/MyTosWikiFetcher，market 以「上市」開頭，含創新板）、history.csv.gz（stock_list=TWSE；目前市場別，最弱）。
  OT（上櫃）：Tasi 淨值「櫃」檔、Tasi 營收月檔上櫃段、bacardi market=TPEx、ISIN 上櫃快照、history（stock_list=TPEX）。
  DL（下市）：o_infra_v3_rebuild/delist_official.csv 的終止上市日。
  第三方對照（不參與判定）：tw_revenue.csv 的 market（2013 年起為「目前市場別」，只報一致率）。
判定（每個代碼、每個交易日 d；同日多筆證據取優先序最高者）：p＝d 之前（含）最近證據、n＝d 之後（含）最近證據：
  p＝TW 且 n∈{TW, 無, OT} → 上市（TW→OT 轉出：到最後一個 TW 證據為止仍算上市，之後到 OT 證據之間列入核對）；
  p＝OT 或 DL → 非上市；p＝無、n＝TW → 從上市日起算（ISIN 上市日；沒有則用第一個 TW 證據日；之前的成交不收，列入核對）。
  上市期間內沒有成交的日子（停牌）仍標上市。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7common as C, srcload7 as S

MANUAL = json.loads((C.HERE / 'listing_manual.json').read_text(encoding='utf-8')) if (C.HERE / 'listing_manual.json').exists() else {}
PRIO = {'panel_official': 9, 'bacardi': 8, 'bacardi_tpex': 8, 'leochen': 7, 'isin': 7, 'tasi_rev': 6, 'tasi_pbr': 5, 'panel_src3': 4, 'history_snapshot': 1, 'delist': 10}

cl6 = pd.read_parquet(C.V6DATA / 'close_raw.parquet'); src6 = pd.read_parquet(C.V6DATA / 'src.parquet')
IDX = cl6.index; COLS = list(cl6.columns)
cal = C.taiex_cal()
TD = IDX.union(cal[(cal >= '2004-01-01') & (cal <= '2026-10-27')])

def is_common(c): return isinstance(c, str) and len(c) == 4 and c.isdigit() and c[0] != '0'

STRONG = len(sys.argv) > 1 and sys.argv[1] == 'strong'   # 只用逐日官方行情類證據（面板官方、leochen、bacardi、ISIN、delist）
def evidence():
    rows = []
    for codes_src, mk in (([1, 2, 4], 'panel_official'), ([3], 'panel_src3')):
        st = (cl6.notna() & src6.isin(codes_src)).stack(); st = st[st]
        rows.append(pd.DataFrame({'date': st.index.get_level_values(0), 'code': st.index.get_level_values(1), 'mkt': 'TW', 'src': mk}))
    for nm, mk in (('leochen', 'TW'), ('bacardi', 'TW'), ('bacardi_tpex', 'OT')):
        d = S.get(nm); rows.append(pd.DataFrame({'date': d.date, 'code': d.code, 'mkt': mk, 'src': nm}))
    d = S.get('tasi_pbr'); rows.append(pd.DataFrame({'date': d.date, 'code': d.code, 'mkt': np.where(d.mkt == 'TWSE', 'TW', 'OT'), 'src': 'tasi_pbr'}))
    r = S.get('tasi_rev'); r = r[r.pit]          # 只用當時加入的月檔（PIT），證據日＝檔案加入日
    rows.append(pd.DataFrame({'date': r.added, 'code': r.code, 'mkt': np.where(r.mkt == 'TWSE', 'TW', 'OT'), 'src': 'tasi_rev'}))
    I = S.get('isin')
    I2 = I[I.file.str.startswith('isin_02') & I.market.str.startswith('上市') & I.group.isin(['股票', '普通股'])]
    rows.append(pd.DataFrame({'date': I2.version, 'code': I2.code.str.strip(), 'mkt': 'TW', 'src': 'isin'}))
    I4 = I[I.file.str.startswith('isin_04') & (I.market == '上櫃') & (I.group == '股票')]
    rows.append(pd.DataFrame({'date': I4.version, 'code': I4.code.str.strip(), 'mkt': 'OT', 'src': 'isin'}))
    sl = pd.read_csv(C.SLIST, dtype=str); slm = dict(zip(sl.code, sl.market))
    H = S.get('history'); hm = H.code.map(slm).map({'TWSE': 'TW', 'TPEX': 'OT'})
    rows.append(pd.DataFrame({'date': H.date, 'code': H.code, 'mkt': hm, 'src': 'history_snapshot'}).dropna())
    dl = pd.read_csv(C.AG / 'o_infra_v3_rebuild' / 'delist_official.csv', dtype=str)
    rows.append(pd.DataFrame({'date': pd.to_datetime(dl.date), 'code': dl.code, 'mkt': 'DL', 'src': 'delist'}))
    E = pd.concat(rows, ignore_index=True); E['code'] = E.code.astype(str).str.strip()
    E = E[E.code.map(is_common)]
    E['p'] = E.src.map(PRIO)
    if STRONG: E = E[E.src.isin(['panel_official', 'leochen', 'bacardi', 'bacardi_tpex', 'isin', 'delist'])]
    return E

def trades():
    ts = []
    t = cl6.notna().stack(); t = t[t]
    ts.append(pd.DataFrame({'date': t.index.get_level_values(0), 'code': t.index.get_level_values(1), 'src': 'panel'}))
    for nm in ('hschi', 'tasi', 'bacardi', 'bacardi_tpex', 'leochen', 'history'):
        d = S.get(nm); ts.append(pd.DataFrame({'date': d.date, 'code': d.code, 'src': nm}))
    T = pd.concat(ts, ignore_index=True)
    return T[T.code.map(is_common)]

def status_days(ev, days, start):
    """ev：已去同日重複的 (date→mkt) Series（排序）。回傳 days 上的 bool（上市）與分類字串陣列。"""
    evd = ev.index.values.astype('datetime64[ns]'); evm = ev.values
    dd = days.values.astype('datetime64[ns]')
    kp = np.searchsorted(evd, dd, side='right') - 1
    kn = np.searchsorted(evd, dd, side='left')
    p = np.where(kp >= 0, evm[np.clip(kp, 0, None)], None)
    n = np.where(kn < len(evm), evm[np.clip(kn, None, len(evm) - 1)], None)
    lab = np.full(len(dd), '?', dtype=object)
    lab[(p == 'TW') & ((n == 'TW') | (n == None))] = 'TW'
    lab[(p == 'TW') & (n == 'OT')] = 'TW>OT'
    lab[(p == 'TW') & (n == 'DL')] = 'TW'
    lab[(p == 'OT')] = 'OT'
    lab[(p == 'DL')] = 'DL'
    lab[(p == None) & (n == 'TW')] = 'pre'
    lab[(dd >= np.datetime64(start)) & (p == None) & (n == 'TW')] = 'pre'
    lab[(p == None) & (n != 'TW')] = 'none'
    listed = np.isin(lab, ['TW', 'TW>OT']) | ((lab == 'pre') & (dd >= np.datetime64(start)))
    return listed, lab

def main():
    E = evidence(); T = trades()
    I = S.get('isin'); I2 = I[I.file.str.startswith('isin_02') & I.market.str.startswith('上市') & I.group.isin(['股票', '普通股'])]
    isin_listed = {c.strip(): pd.Timestamp(x.replace('/', '-')) for c, x in zip(I2.code, I2.listed)}
    isin_mkt = {c.strip(): m for c, m in zip(I2.code, I2.market)}
    rev = pd.read_csv(C.REV, dtype=str); rev['month'] = pd.to_datetime(rev.month)
    trv = S.get('tasi_rev')
    codes = sorted(set(COLS) | set(E[E.mkt == 'TW'].code))
    Eg = {c: g for c, g in E.groupby('code')}; Tg = {c: g for c, g in T.groupby('code')}
    rows, review = [], []
    LMcols = {}
    tdays_all = pd.DatetimeIndex(TD)
    for c in codes:
        e = Eg.get(c)
        if e is None or not (e.mkt == 'TW').any(): continue
        t = Tg.get(c)
        tdays = pd.DatetimeIndex(sorted(set(t.date))) if t is not None else pd.DatetimeIndex([])
        # 同日兩市（TW 與 OT 同時出現）→ 核對
        nn = e[e.mkt.isin(['TW', 'OT'])].groupby('date').mkt.nunique(); both = nn[nn > 1]
        if len(both):
            sub = e[e.date.isin(both.index)]
            review.append({'code': c, 'issue': 'same_day_both_markets', 'n': int(len(both)), 'first': str(both.index.min().date()), 'last': str(both.index.max().date()),
                           'srcs': {f'{a}|{b}': int(v) for (a, b), v in sub.groupby(['mkt', 'src']).size().items()}})
        ee = e.sort_values(['date', 'p']).groupby('date').tail(1)
        ev = ee.set_index('date').mkt.sort_index()
        tw = ev[ev == 'TW']; first_tw, last_tw = tw.index.min(), tw.index.max()
        man = MANUAL.get(c, {})
        if man.get('start'): start, ssrc = pd.Timestamp(man['start']), 'manual'
        elif c in isin_listed and not (ev[ev.index < isin_listed[c]].isin(['OT', 'DL'])).any() and isin_listed[c] >= pd.Timestamp('2004-01-01'):
            start, ssrc = isin_listed[c], 'isin_listedDate'      # ISIN 上市日為準（早於或晚於第一個 TW 證據都用它）
        elif c in isin_listed and isin_listed[c] < pd.Timestamp('2004-01-01'):
            start, ssrc = isin_listed[c], 'isin_listedDate'
        else:
            # 起點只用逐日行情類證據（面板官方、Yahoo .TW、leochen、bacardi、淨值檔）；月檔與 history 快照不定起點
            twd = e[(e.mkt == 'TW') & e.src.isin(['panel_official', 'panel_src3', 'leochen', 'bacardi', 'tasi_pbr', 'isin'])].date
            start, ssrc = (twd.min() if len(twd) else first_tw), 'first_TW_evidence'
        if ssrc == 'isin_listedDate':
            ev = ev[~((ev.index < start) & (ev == 'TW'))]           # ISIN 上市日之前的 TW 證據（非 PIT 月檔等）不用
            tw = ev[ev == 'TW']; first_tw = tw.index.min()
        listed_td, lab_td = status_days(ev, tdays_all, start)
        if man.get('end'): listed_td &= tdays_all <= pd.Timestamp(man['end'])
        for a, b in man.get('exclude', []): listed_td &= ~((tdays_all >= pd.Timestamp(a)) & (tdays_all <= pd.Timestamp(b)))
        for a, b in man.get('include', []): listed_td |= ((tdays_all >= pd.Timestamp(a)) & (tdays_all <= pd.Timestamp(b)))
        # 上市期間之後沒有成交（下市）：最後一個「上市且有成交」日之後，若沒有 TW 證據，截斷（停牌延續到下市日由 DL 證據決定）
        lst = pd.Series(listed_td, index=tdays_all)
        trd = lst.index.isin(tdays)
        if len(tdays):
            last_tr = lst[lst & trd].index.max() if (lst & trd).any() else None
            if last_tr is not None:
                cut = lst.index > max(last_tr, last_tw)
                dl = e[e.mkt == 'DL'].date
                dlx = dl[dl > last_tr]
                if len(dlx): cut = lst.index >= dlx.min()
                lst[cut] = False
        LMcols[c] = lst.to_numpy()
        # 核對項：上市日前有成交（其他來源）
        lab_tr = pd.Series(lab_td, index=tdays_all).reindex(tdays)
        pre_tr = tdays[(lab_tr == 'pre').to_numpy() & (tdays < start)]
        if len(pre_tr) and start >= pd.Timestamp('2019-06-01'):
            review.append({'code': c, 'issue': 'trades_before_listing_start', 'n_days': int(len(pre_tr)), 'first_trade': str(pre_tr.min().date()), 'start': str(start.date()),
                           'start_src': ssrc, 'srcs_before': {k: int(v) for k, v in t[t.date < start].groupby('src').size().items()}})
        ot_tr = tdays[(lab_tr == 'OT').to_numpy() & (tdays >= pd.Timestamp('2019-06-01'))]
        twot = tdays[(lab_tr == 'TW>OT').to_numpy() & (tdays >= pd.Timestamp('2019-06-01'))]
        if len(twot):
            review.append({'code': c, 'issue': 'TW_then_OT', 'n_days': int(len(twot)), 'from': str(twot.min().date()), 'to': str(twot.max().date())})
        # 轉入：OT 證據早於 TW 證據
        otb = ev[(ev == 'OT')]
        trans = None
        if len(otb) and (otb.index < last_tw).any():
            ot_last = otb[otb.index < last_tw].index.max(); nxt = tw[tw.index > ot_last].index.min()
            trans = nxt
            review.append({'code': c, 'issue': 'OT_to_TW_transfer', 'last_OT': str(ot_last.date()), 'first_TW_after': str(nxt.date()),
                           'gap_trading_days': int(((tdays_all > ot_last) & (tdays_all < nxt)).sum()), 'n_OT_trade_days_2019on': int(len(ot_tr))})
        L = lst[lst]
        segs = []
        if len(L):
            idx = np.where(lst.to_numpy())[0]; br = np.where(np.diff(idx) > 1)[0]
            st_i = np.r_[idx[0], idx[br + 1]]; en_i = np.r_[idx[br], idx[-1]]
            segs = [(str(tdays_all[a].date()), str(tdays_all[b].date())) for a, b in zip(st_i, en_i)]
        # 第三方對照：tw_revenue market 與 Tasi 營收段（以次月 10 日的上市狀態比）
        rv = rev[rev.code == c]
        agree_rev = None; n_rev = 0
        if len(rv):
            m = rv[(rv.month >= '2013-01-01')]
            dts = m.month + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)
            s_at = lst.reindex(lst.index.union(dts)).ffill().reindex(dts).fillna(False).to_numpy()
            ok = (m.market.to_numpy() == 'sii') == s_at
            agree_rev = float(ok.mean()) if len(ok) else None; n_rev = int(len(ok))
        tr_ = trv[trv.code == c]
        agree_trv = None; n_trv = 0
        if len(tr_):
            dts = (pd.PeriodIndex(tr_.ym, freq='M') + 1).to_timestamp() + pd.Timedelta(days=9)
            s_at = lst.reindex(lst.index.union(dts)).ffill().reindex(dts).fillna(False).to_numpy()
            ok = (tr_.mkt.to_numpy() == 'TWSE') == s_at
            agree_trv = float(ok.mean()); n_trv = int(len(ok))
        rows.append({'code': c, 'in_panel_cols': c in set(COLS), 'listed_date': str(start.date()), 'listed_src': ssrc,
                     'isin_listed': str(isin_listed[c].date()) if c in isin_listed else '', 'isin_market': isin_mkt.get(c, ''),
                     'transfer_in_date': str(trans.date()) if trans is not None else '',
                     'delist_date': ';'.join(str(x.date()) for x in e[e.mkt == 'DL'].date),
                     'segments': ';'.join(f'{a}~{b}' for a, b in segs),
                     'first_listed_day': segs[0][0] if segs else '', 'last_listed_day': segs[-1][1] if segs else '',
                     'first_TW_evidence': str(first_tw.date()), 'last_TW_evidence': str(last_tw.date()),
                     'evidence_srcs': ';'.join(sorted(set(e.src))),
                     'agree_tw_revenue_market': agree_rev, 'n_tw_revenue_months': n_rev,
                     'agree_tasi_rev_section': agree_trv, 'n_tasi_rev_months': n_trv,
                     'n_listed_days': int(lst.sum()), 'manual_note': man.get('note', '')})
    M = pd.DataFrame(rows)
    rv_ids = {r['code'] for r in review}
    M['check_status'] = np.where(M.code.isin(MANUAL.keys()), 'manual_verified', np.where(M.code.isin(rv_ids), 'flagged', 'consistent'))
    if STRONG:
        allc = list(COLS) + [c for c in LMcols if c not in set(COLS)]
        pd.DataFrame({c: LMcols.get(c, np.zeros(len(TD), bool)) for c in allc}, index=TD).to_parquet(C.DATA / 'listed_twse_strong.parquet'); print('strong written'); return
    M.to_csv(C.HERE / 'listing_master.csv', index=False)
    allc = list(COLS) + [c for c in LMcols if c not in set(COLS)]
    LM = pd.DataFrame({c: LMcols.get(c, np.zeros(len(TD), bool)) for c in allc}, index=TD)
    LM.to_parquet(C.DATA / 'listed_twse.parquet')
    C.jdump({'n_codes': len(M), 'n_in_panel_cols': int(M.in_panel_cols.sum()), 'status_counts': M.check_status.value_counts().to_dict(),
             'issue_counts': pd.Series([r['issue'] for r in review]).value_counts().to_dict() if review else {}, 'review': review}, C.DATA / 'listing_review.json')
    print(M.check_status.value_counts().to_dict(), pd.Series([r['issue'] for r in review]).value_counts().to_dict())

if __name__ == '__main__':
    main()
