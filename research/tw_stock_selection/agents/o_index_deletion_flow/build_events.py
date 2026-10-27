"""建立 events_v2.csv（事件名單；只用名單與交易日曆，不碰任何報酬）。
輸入：lists_transcribed.tsv（人工從新聞抄錄的 MSCI 增刪，每列附來源）、../etf_rebalance/events.csv（臺灣50／0056／00878／00919）、
      hd_etf_transcribed.tsv（新增高股息 ETF，若存在）。輸出：events_v2.csv、events_v2_unmatched.csv。"""
import sys, os, re, hashlib, json
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import namemap as nm
HERE = os.path.dirname(os.path.abspath(__file__))
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
cal = pd.read_parquet(P + 'close_raw.parquet', columns=['1101']).index  # 交易日曆
panel_codes = set(pd.read_parquet(P + 'src.parquet').columns)
d = nm.build()

def trade_day_on_or_before(s):
    t = pd.Timestamp(s); i = cal.searchsorted(t, side='right') - 1
    return cal[i]

rows, unmatched = [], []
L = pd.read_csv(os.path.join(HERE, 'lists_transcribed.tsv'), sep='\t', dtype=str).fillna('')
for rv, g in L.groupby('review'):
    std_add = set(); std_del = set(); sc_add = set(); sc_del = set()
    def names(s): return [nm.norm(x) for x in re.split('、', s) if x.strip() and not x.startswith('(')]
    for r in g.itertuples():
        S = set(names(r.names))
        if r.index == 'MSCI_STD' and r.action == 'add': std_add |= S
        if r.index == 'MSCI_STD' and r.action == 'del': std_del |= S
        if r.index == 'MSCI_SC' and r.action == 'add': sc_add |= S
        if r.index == 'MSCI_SC' and r.action == 'del': sc_del |= S
    for r in g.itertuples():
        eff_stated = r.eff_stated
        E = trade_day_on_or_before(eff_stated)
        for nmx in names(r.names):
            code = d.get(nmx)
            if r.index == 'MSCI_STD' and r.action == 'del':
                if 'full_exit' in r.note: sub = 'STD_full_exit'
                elif nmx in sc_add or 'to_SC' in r.note: sub = 'STD_to_SC'
                else: sub = 'STD_unverified'
                typ = 'MSCI_STD_DEL'
            elif r.index == 'MSCI_STD' and r.action == 'add':
                typ = 'MSCI_STD_ADD'; sub = 'from_SC' if (nmx in sc_del or 'from_SC' in r.note) else 'new_or_unverified'
            elif r.index == 'MSCI_SC' and r.action == 'del':
                if nmx in std_add: continue  # 升級到標準指數，不是刪除
                typ = 'MSCI_SC_DEL'; sub = 'SC_exit'
            else:
                if nmx in std_del: continue  # 從標準降級，已在 STD_DEL 記錄
                typ = 'MSCI_SC_ADD'; sub = 'SC_new'
            rec = dict(family='MSCI', index=r.index, review=rv, name=nmx, code=code or '', type=typ, sub=sub,
                       eff_stated=eff_stated, eff=E.date().isoformat(), verify='eff_stated_in_source', src=r.src, note=r.note)
            if not code: unmatched.append(rec); continue
            rows.append(rec)
# 既有 ETF 事件
E0 = pd.read_csv(os.path.join(HERE, '..', 'etf_rebalance', 'events.csv'), dtype=str).fillna('')
for r in E0.itertuples():
    typ = f'{r.etf}_{"DEL" if r.action=="del" else "ADD"}'
    E = trade_day_on_or_before(r.eff)
    rows.append(dict(family='ETF', index=r.etf, review=r.review, name=r.name, code=r.code, type=typ, sub='',
                     eff_stated=r.eff, eff=E.date().isoformat(), verify='etf_rebalance:' + r.verify, src=r.src, note=r.note))
hd = os.path.join(HERE, 'hd_etf_transcribed.tsv')
if os.path.exists(hd):
    H = pd.read_csv(hd, sep='\t', dtype=str).fillna('')
    for r in H.itertuples():
        for nmx in [nm.norm(x) for x in r.names.split('、') if x.strip()]:
            code = d.get(nmx); E = trade_day_on_or_before(r.eff_stated)
            rec = dict(family='ETF', index=r.etf, review=r.review, name=nmx, code=code or '', type=f'{r.etf}_{"DEL" if r.action=="del" else "ADD"}',
                       sub='', eff_stated=r.eff_stated, eff=E.date().isoformat(), verify=r.verify, src=r.src, note=r.note)
            if not code: unmatched.append(rec); continue
            rows.append(rec)
ev = pd.DataFrame(rows)
ev['in_panel'] = ev.code.isin(panel_codes)
ev['src3_period'] = (pd.to_datetime(ev.eff) >= '2024-02-15') & (pd.to_datetime(ev.eff) <= '2025-09-23')
ev = ev.drop_duplicates(['type', 'code', 'eff']).sort_values(['eff', 'type', 'code']).reset_index(drop=True)
ev.insert(0, 'event_id', [f'E{i:04d}' for i in range(len(ev))])
ev.to_csv(os.path.join(HERE, 'events_v2.csv'), index=False)
pd.DataFrame(unmatched).to_csv(os.path.join(HERE, 'events_v2_unmatched.csv'), index=False)
print(len(ev), 'events;', len(unmatched), 'unmatched names')
print(ev[ev.in_panel].groupby('type').size())
