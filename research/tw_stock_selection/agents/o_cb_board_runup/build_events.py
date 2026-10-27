# -*- coding: utf-8 -*-
"""第 0 步：建事件表（不計算任何報酬）。
錨點：MOPS 重大訊息「董事會決議發行（國內）轉換公司債」的發言日期（當時即公開）。
放回撤回／退件／撤銷／從未送件的案件：事件表不以事後結果篩選，只把結果當標記。
另建安慰劑事件：普通公司債董事會決議（SB）、私募 CB 董事會決議（PCB）；海外 CB（ECB）只當子群組。
輸出：events_board.csv、build_summary.json（含覆蓋率、閘門 (a)）。"""
import re, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import cblib as L

HERE = L.HERE
T = pd.read_parquet(HERE / 'mops_titles.parquet')
T['ts'] = T['date']
CBW = r'轉換公司債|可轉換公司債|可轉債|轉換債'
BOARD = r'董事會|審計委員會|擬發行|擬辦理|擬募集'
EXC_COMMON = (r'基準日|暫停|停止|調整|重設|賣回|買回|贖回|償還|還本|付息|下櫃|終止|代收|存儲|專戶|延長|展延|執行情形|資金運用|'
              r'發行新股|全數轉換|註銷|收足|募集完成|撤銷|撤回|價格訂定|訂價|定價|轉換價格|利率|子公司|代.{0,12}公司公告|上櫃|掛牌|'
              r'股東會|轉換普通股|換發|轉換之|轉換為')

t = T['title']
is_cb = t.str.contains(CBW)
# 標準格式的發行公告主旨常常沒有「董事會」字樣（例：「發行國內第三次無擔保可轉換公司債」），也算董事會決議公告
std = t.str.match(r'^(公告)?(本公司)?(擬)?(發行|募集)?.{0,10}(國內|境內)?.{0,10}(無擔保|有擔保|擔保).{0,3}(可)?轉換公司債(案|相關事宜|之發行)?[。.]?$')
is_board = t.str.contains(BOARD) | std
exc = t.str.contains(EXC_COMMON)
priv = t.str.contains('私募')
over = t.str.contains('海外') & ~t.str.contains('國內|境內')
T['kind'] = None
T.loc[is_cb & is_board & ~exc & ~priv & ~over, 'kind'] = 'CB'
T.loc[is_cb & is_board & ~exc & ~priv & over, 'kind'] = 'ECB'
T.loc[is_cb & is_board & ~exc & priv, 'kind'] = 'PCB'
sb = (t.str.contains('公司債') & ~is_cb & t.str.contains(BOARD) & ~t.str.contains(EXC_COMMON) & ~priv
      & ~t.str.contains('海外|金融債|次順位|永續|特別股'))
T.loc[sb, 'kind'] = 'SB'
# 定價：國內 CB 完成訂價／訂價基準日公告
pr = is_cb & t.str.contains('訂價|定價|轉換價格訂定|轉換價格.{0,4}基準日') & ~t.str.contains('調整|重設|私募|海外|停止|暫停|修正')
T.loc[pr & T['kind'].isna(), 'kind'] = 'PRICE'
K_ = T[T['kind'].notna()].sort_values(['code', 'kind', 'date', 'tm']).copy()

# 去重：同公司同類型，距上一則同類公告 > 120 個日曆日才算新事件；事件日＝群組第一則
ev = []
for (c, k), g in K_[K_.kind != 'PRICE'].groupby(['code', 'kind']):
    last = None; cur = None
    wd = T[(T.code == c) & T.title.str.contains('撤回|撤銷') & T.title.str.contains('公司債')]['date']
    for _, r in g.iterrows():
        split_wd = last is not None and ((wd > last) & (wd < r['date'])).any()
        if last is None or (r['date'] - last).days > 90 or split_wd:
            if cur is not None: ev.append(cur)
            cur = dict(code=c, kind=k, board=r['date'], tm=r['tm'], title=r['title'], n_ann=1, name=r['name'],
                       combo=bool(re.search('現金增資|普通股|存託憑證', r['title'])))
        else:
            cur['n_ann'] += 1
        last = r['date']
    if cur is not None: ev.append(cur)
E = pd.DataFrame(ev)
PRICE = K_[K_.kind == 'PRICE'][['code', 'date', 'title']]

# ---------- FSC 申報案件（含撤回、退件、撤銷） ----------
F = pd.read_csv(HERE.parent / 'o_issuance_avoid' / 'fsc_cases_all.csv', dtype={'code': str}, parse_dates=['recv', 'eff'])
F = F[(F.ctype == '上市') & F['item'].astype(str).str.contains('轉換') & ~F['item'].astype(str).str.contains('海外|私募|金融債')].copy()
F['withdrawn'] = F['result'] != '生效'
FCOV = json.load(open(HERE.parent / 'o_issuance_avoid' / 'events_build_summary.json'))['fsc_year_coverage']
def fsc_covered(d):
    y = str(d.year)
    if y not in FCOV: return False
    a, b = FCOV[y]
    return pd.Timestamp(a) <= d <= pd.Timestamp(b)

TP = pd.read_csv(HERE.parent / 'o_issuance_avoid' / 'tpex_cb_listings.csv', dtype={'cb_code': str, 'issuer': str}, parse_dates=['list_date'])

rows = []
used_f = set()
for i, r in E.iterrows():
    d = r['board']; c = r['code']
    o = dict(fsc_recv=pd.NaT, fsc_result=None, fsc_amount=np.nan, list_date=pd.NaT, cb_code=None, price_date=pd.NaT)
    if r['kind'] == 'CB':
        f = F[(F.code == c) & (F.recv >= d - pd.Timedelta(days=15)) & (F.recv <= d + pd.Timedelta(days=240))].sort_values('recv')
        if len(f):
            ff = f.iloc[0]; used_f.add(ff.name)
            o.update(fsc_recv=ff['recv'], fsc_result=ff['result'], fsc_amount=ff['amount'])
        tp = TP[(TP.issuer == c) & (TP.list_date >= d) & (TP.list_date <= d + pd.Timedelta(days=365))].sort_values('list_date')
        if len(tp):
            o.update(list_date=tp.iloc[0]['list_date'], cb_code=tp.iloc[0]['cb_code'])
        p = PRICE[(PRICE.code == c) & (PRICE.date >= d) & (PRICE.date <= (o['list_date'] if pd.notna(o['list_date']) else d + pd.Timedelta(days=365)))]
        if len(p): o['price_date'] = p['date'].min()
    rows.append(o)
E = pd.concat([E, pd.DataFrame(rows)], axis=1)

def outcome(r):
    if r['kind'] != 'CB': return None
    if pd.notna(r['list_date']): return 'listed'
    if pd.notna(r['fsc_result']) and r['fsc_result'] != '生效': return 'withdrawn'
    if pd.notna(r['fsc_result']) and r['fsc_result'] == '生效': return 'effective_unlisted'
    return 'no_filing' if fsc_covered(r['board']) else 'unknown_fsc_gap'
E['outcome'] = E.apply(outcome, axis=1)

# 定價日 → 掛牌日落差（交易日），用來在沒有定價公告時推估基準日
lag = []
for _, r in E[(E.kind == 'CB') & E.price_date.notna() & E.list_date.notna()].iterrows():
    a = L.tpos_on_after(r['price_date']); b = L.tpos_on_after(r['list_date'])
    if a is not None and b is not None: lag.append(b - a)
LAG = int(np.median(lag)) if lag else 6
def base_idx(r):
    if pd.notna(r['price_date']): return L.tpos_on_after(r['price_date'])
    if pd.notna(r['list_date']):
        b = L.tpos_on_after(r['list_date']); return b - LAG if b is not None else None
    return None

# 面板特徵（公告日前一交易日已知）
recs = []
for _, r in E.iterrows():
    j = L.CI.get(r['code'])
    t1 = L.tpos_after(r['board'])                         # 進場日＝公告日後第一個交易日（開盤）
    t0 = L.tpos_on_before(r['board'])                     # 公告日當天或之前最後一個交易日（特徵用）
    x = dict(j=j, t_entry=t1, entry_date=L.IDX[t1] if t1 is not None else pd.NaT)
    if j is not None and t0 is not None:
        x.update(adv60=L.V60[t0, j], px=L.Cn[t0, j], s52=L.S52[t0, j], fin=bool(L.FINMASK[j]),
                 traded_entry=bool(t1 is not None and np.isfinite(L.On[t1, j])), in_pool300=bool(L.INPOOL[t0, j]))
        # 市值近似不可得（無股數）：以 60 日均值當規模代理
    if r['kind'] == 'CB':
        bi = base_idx(r)
        x['t_base'] = bi
        x['h_board_base'] = (bi - t1 + 1) if (bi is not None and t1 is not None) else np.nan   # 進場日到基準日的交易日數（含兩端）
        bl = L.tpos_on_after(r['list_date']) if pd.notna(r['list_date']) else None
        x['t_list'] = bl
        x['h_board_list'] = (bl - t1 + 1) if (bl is not None and t1 is not None) else np.nan
        x['d_board_recv'] = (r['fsc_recv'] - r['board']).days if pd.notna(r['fsc_recv']) else np.nan
    recs.append(x)
E = pd.concat([E, pd.DataFrame(recs)], axis=1)
# 近 2 年曾在 TPEx 掛牌 CB（重複發行者；只用公告日前已知的掛牌）
def repeat(r):
    p = TP[(TP.issuer == r['code']) & (TP.list_date < r['board']) & (TP.list_date >= r['board'] - pd.Timedelta(days=730))]
    return bool(len(p))
E['repeat2y'] = E.apply(repeat, axis=1)
E = E.sort_values(['board', 'code']).reset_index(drop=True)
E.insert(0, 'event_id', [f'B{i:05d}' for i in range(len(E))])
E.to_csv(HERE / 'events_board.csv', index=False)

# ---------- 覆蓋率：有董事會日的 FSC 上市國內 CB 案件 ÷ FSC 上市國內 CB 案件（依收文年） ----------
SE = set(pd.read_csv(HERE / 'mops_SE_list.csv', dtype=str)['code'])
F['matched'] = F.index.isin(used_f)
F['in_mops_list'] = F['code'].isin(SE)
cov = F[(F.recv >= '2011-01-01') & (F.recv <= '2020-08-31')].groupby(F.recv.dt.year).agg(
    fsc_cases=('code', 'size'), matched=('matched', 'sum'), in_mops_list=('in_mops_list', 'sum'), withdrawn=('withdrawn', 'sum'),
    withdrawn_matched=('matched', lambda s: int((s & F.loc[s.index, 'withdrawn']).sum())))
cov['coverage'] = cov['matched'] / cov['fsc_cases']
cov['coverage_of_survivors'] = cov['matched'] / cov['in_mops_list']

# ---------- 閘門 (a) ----------
cb = E[E.kind == 'CB']
sel = cb[(cb.board >= '2010-01-01') & (cb.board <= '2018-12-31')]
base = sel[sel.j.notna() & sel.traded_entry.fillna(False).astype(bool)]
liq = base[base.adv60 >= 3e7]
liq_nf = liq[~liq.fin.astype(bool)]
gate_a = dict(rule='挑選期 2010～2018 有董事會日錨點、上市、60 日均成交值 ≥ 3,000 萬的國內 CB 事件 ≥ 200（含撤回／未送件；排除金融後計數為正式數）',
              n_board_events_2010_2018=int(len(sel)), n_in_panel_tradable=int(len(base)), n_liquid=int(len(liq)), n_liquid_exfin=int(len(liq_nf)),
              n_liquid_exfin_by_outcome=liq_nf['outcome'].value_counts().to_dict(),
              n_liquid_exfin_by_year=liq_nf.groupby(liq_nf.board.dt.year).size().to_dict(),
              n_liquid_exfin_incl_ECB=int(len(liq_nf)) + int(((E.kind == 'ECB') & (E.board >= '2010-01-01') & (E.board <= '2018-12-31') & (E.adv60 >= 3e7) & ~E.fin.fillna(True).astype(bool)).sum()),
              passed=bool(len(liq_nf) >= 200))
# ---------- 閘門 (a) 的上界：不論董事會日來源多完整，最多能有幾筆 ----------
# FSC 上市國內 CB 申報（2010～2018 收文，含撤回）中，收文日前一交易日 60 日均值 ≥ 3,000 萬、非金融者
#   ＋ 已有 MOPS 董事會日、但 FSC 沒有對應申報（未送件，或 FSC 缺年 2016／2009～2010、2013 部分缺月）的事件
ub = []
for _, f in F[(F.recv >= '2010-01-01') & (F.recv <= '2018-12-31')].iterrows():
    j = L.CI.get(f['code']); t0 = L.tpos_on_before(f['recv'])
    if j is None or t0 is None: continue
    ub.append(dict(code=f['code'], recv=f['recv'], adv60=L.V60[t0, j], fin=bool(L.FINMASK[j]), matched=bool(f.name in used_f)))
UB = pd.DataFrame(ub)
ub_fsc = UB[(UB.adv60 >= 3e7) & ~UB.fin]
nofsc = liq_nf[liq_nf.fsc_recv.isna()]
gate_a['upper_bound'] = dict(fsc_liquid_exfin=int(len(ub_fsc)), fsc_liquid_exfin_by_year=ub_fsc.groupby(ub_fsc.recv.dt.year).size().to_dict(),
                             mops_events_without_fsc=int(len(nofsc)), total=int(len(ub_fsc) + len(nofsc)),
                             note='2016 FSC 整年缺、2009-09～2010-01 與 2010-08～12、2013-12 缺；這些月份只能靠 MOPS（2011 起）補，2010 年無任何董事會日來源')
summ = dict(source='MOPS 重大訊息標題（shinbutou/nccu_bigdata，上市公司 2011-01～2020-10，932 檔，2020 年存活清單）',
            n_events=E.kind.value_counts().to_dict(), cb_by_year=cb.groupby(cb.board.dt.year).size().to_dict(),
            cb_outcome=cb.outcome.value_counts().to_dict(), price_announcements=int(len(PRICE)), price_to_list_lag_td=dict(median=LAG, n=len(lag), q=np.percentile(lag, [10, 25, 75, 90]).tolist() if lag else None),
            coverage_by_recv_year=cov.reset_index().to_dict('records'),
            coverage_total=dict(fsc=int(cov.fsc_cases.sum()), matched=int(cov.matched.sum()), in_mops_list=int(cov.in_mops_list.sum()),
                                withdrawn=int(cov.withdrawn.sum()), withdrawn_matched=int(cov.withdrawn_matched.sum())),
            board_to_recv_days=dict(median=float(cb.d_board_recv.median()), q=cb.d_board_recv.quantile([.1, .25, .75, .9]).tolist(), n=int(cb.d_board_recv.notna().sum())),
            gate_a=gate_a, note_2010='MOPS 鏡像自 2011-01 起；2010 年無董事會日來源（新聞鏡像 2013 起），2010 事件數為 0。')
L.dump(summ, 'build_summary.json')
print(json.dumps(summ, ensure_ascii=False, indent=1, default=L.conv)[:6000])
