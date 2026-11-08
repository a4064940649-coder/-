#!/usr/bin/env python3
"""獨立複製（第二位研究員）之一：只從 ../raw/ev_*.txt 證據摘錄重建對價表，不 import 研究員任何模組（cz_*、s*_*.py）。
母體＝普查到的被持有下市事件（../census.json 的 distinct_perm_events_U ∪ distinct_perm_events_IA；普查本身是資料產品，未重寫）。
自寫規則：
  數字只從摘錄的「關鍵句」抽：現金（「現金 X 元」「X元現金」「每股X元」「'X元'」）、比例（「X 股」且不是 1；多公司共用句時取含該公司名的子句）、
  特別股（「X 股特別股」與「發行價格 X 元」）、「X股換1股」→ 1/X。
  型態：有比例＋現金→mixed；有比例且句中有「控股／投控／股份轉換／轉換」→conversion；有比例→swap；只有現金→cash；
       無數字但有「合併／併入／存續／併了／轉給」且點名存續公司→swap_ur；無數字但有「收購／子公司／品牌」→acq_ur；無證據→unknown。
  有效比例核實：純換股／轉換若 lt 當日市場隱含價值偏離 > 5%、存續公司 lt 前 250 交易日有除權（flag 1），且摘錄沒有「除權息」字樣 → swap_ur。
  存續公司代碼：名稱表（stock_list.csv）＋歷史別名（開發金→2883、新設控股公司→終止日當天首日掛牌且名稱含標的名者）。
輸出 rp_events.csv。python3 -I -B rp_events.py"""
import re, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TOP = HERE.parent
AG = TOP.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
SL = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str)
ALIAS = {'開發金': '2883', '中信金': '2891', '元大金': '2885', '台新金': '2887'}
NAME2CODE = dict(zip(SL.name, SL.code)); NAME2CODE.update(ALIAS)
TNAME = {'2847': '大眾銀', '3697': '晨星', '2833': '台壽', '2823': '中國人壽', '2311': '日月光', '2325': '矽品', '3474': '華亞科', '6286': '立錡',
         '1704': '榮化', '1902': '台紙', '2448': '晶電', '3698': '隆達', '2394': '普立爾', '3009': '奇美', '3012': '廣輝', '3063': '飛信', '2452': '乾坤',
         '2473': '思源', '2403': '友尚', '2446': '全懋', '2432': '倚天', '2888': '新光金'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_evidence():
    ev = {}
    for f in sorted((TOP / 'raw').glob('ev_*.txt')):
        code = f.name.split('_')[1]
        t = f.read_text(encoding='utf-8')
        key = re.search(r'^# 關鍵句: (.*)$', t, re.M).group(1)
        body = t.split('\n\n', 1)[1]
        ev.setdefault(code, []).append(dict(file=f.name, sha=sha(f), key=key, body=body))
    return ev


def extract(code, items):
    keys = [x['key'] for x in items]; body = ' '.join(x['body'] for x in items)
    nm = TNAME.get(code, '')
    txt = ' '.join(keys)
    clause = [c for k in keys for c in re.split('[、，,]', k) if nm and nm in c]
    use = ' '.join(clause) if clause and len(clause) < len(keys) * 2 and code in ('2448', '3698') else txt
    o = dict(cash=np.nan, ratio=np.nan, pref_ratio=np.nan, pref_value=np.nan)
    m = re.search(r'([0-9.]+)\s*股換\s*1\s*股', use)
    if m: o['ratio'] = 1 / float(m.group(1))
    m = re.search(r'([0-9.]+)\s*股特別股', use)
    if m: o['pref_ratio'] = float(m.group(1))
    m = re.search(r'發行價格\s*([0-9.]+)\s*元', use)
    if m: o['pref_value'] = float(m.group(1))
    if not np.isfinite(o['ratio']):
        for m in re.finditer(r'([0-9]*\.?[0-9]+)\s*股(?!特別股)', use):
            v = float(m.group(1))
            if v != 1.0 and not re.match(r'\s*股份', use[m.end() - 1:m.end() + 2]):
                o['ratio'] = v; break
    for pat in (r'現金\s*([0-9.]+)\s*元', r'([0-9.]+)\s*元現金', r'每股\s*([0-9.]+)\s*元', r"'([0-9.]+)元'\]", r'([0-9.]+)\s*元現金對價', r'現金\s*([0-9.]+)\s*元對價'):
        m = re.search(pat, use)
        if m and not (o['pref_value'] == float(m.group(1)) and '發行價格' in use[max(0, m.start() - 6):m.start()]):
            o['cash'] = float(m.group(1)); break
    if np.isfinite(o['ratio']) and np.isfinite(o['cash']):
        typ = 'mixed'
    elif np.isfinite(o['ratio']):
        typ = 'conversion' if re.search('控股|投控|股份轉換|轉換', use) else 'swap'
    elif np.isfinite(o['cash']):
        typ = 'cash'
    elif re.search('合併|併入|存續|併了|轉給', txt):
        typ = 'swap_ur'
    elif re.search('收購|子公司|品牌', txt):
        typ = 'acq_ur'
    else:
        typ = 'unknown'
    # 存續公司：句中點名者（排除標的自己）
    succ = ''
    for nmx, c in sorted(NAME2CODE.items(), key=lambda kv: -len(kv[0])):
        if len(nmx) >= 2 and nmx in txt and nmx != nm and not nm.startswith(nmx) and c != code:
            succ = c; break
    if '新設控股公司' in txt:
        succ = 'NEWHOLDCO'
    return typ, o, succ, body


def main():
    C = json.loads((TOP / 'census.json').read_text(encoding='utf-8'))
    pop = {e[0] for e in C['distinct_perm_events_U']} | {e[0] for e in C['distinct_perm_events_IA']}
    # 擴充母體：官方終止上市名單 2006 起四碼＋面板 2024-09-03 之後永久停止交易者（無證據者一律 unknown）
    _dl = pd.read_csv(AG / 'o_infra_v3_rebuild/delist_official.csv', dtype=str)
    pop |= set(_dl[(_dl.four == 'True') & (_dl.date >= '2006-01-01')].code)
    _c7 = pd.read_parquet(AG / 'o_v7_listing_exec_engine/data/close_raw.parquet')
    _lv = _c7.apply(lambda x: x.last_valid_index())
    pop |= set(_lv[(_lv >= '2024-09-03') & (_lv < _c7.index[-1] - pd.Timedelta(days=30))].index)
    pop = sorted(c for c in pop if c in _c7.columns)
    lastU = {e[0]: e[1] for e in C['distinct_perm_events_U']}
    ev = load_evidence()
    # 價格：V7 面板＋下市價修補
    C7 = pd.read_parquet(AG / 'o_v7_listing_exec_engine/data/close_raw.parquet')
    dp = pd.read_parquet(AG / 'o_infra_v3_rebuild/delist_prices.parquet')
    Cp = C7.copy()
    for r in dp.itertuples():
        if pd.isna(Cp.at[r.date, r.code]) if r.date in Cp.index else False: Cp.at[r.date, r.code] = r.close
    FL = pd.read_parquet(SCR / 'panel/flag.parquet')
    DL = pd.read_csv(AG / 'o_infra_v3_rebuild/delist_official.csv', dtype=str); DL = DL[DL.four == 'True']
    rows = []
    for code in pop:
        s = Cp[code].dropna()
        if s.empty: continue
        lt = pd.Timestamp(lastU[code]) if code in lastU else None
        term = DL[DL.code == code].date.tolist(); term = [pd.Timestamp(x) for x in term if x >= '2006-01-01']
        term = term[0] if term else None
        if lt is None:
            s2 = s[s.index <= term] if term is not None else s
            lt = s2.index[-1]
        lc = float(Cp.at[lt, code])
        if code in ev:
            typ, o, succ, body = extract(code, ev[code])
        else:
            typ, o, succ, body = 'unknown', dict(cash=np.nan, ratio=np.nan, pref_ratio=np.nan, pref_value=np.nan), '', ''
        if succ == 'NEWHOLDCO':
            nm = TNAME.get(code, '')
            cand = [c for n, c in NAME2CODE.items() if nm and n.startswith(nm) and n != nm and c in Cp.columns]
            cand = [c for c in cand if term is not None and Cp[c].first_valid_index() is not None and abs((Cp[c].first_valid_index() - term).days) <= 3]
            succ = cand[0] if cand else ''
        dev = np.nan; impl = np.nan
        if succ and succ in Cp.columns and np.isfinite(Cp.at[lt, succ]):
            impl = lc / Cp.at[lt, succ]
            if typ in ('swap', 'conversion', 'mixed'):
                v = o['ratio'] * Cp.at[lt, succ] + (o['cash'] if np.isfinite(o['cash']) else 0) + ((o['pref_ratio'] * o['pref_value']) if np.isfinite(o['pref_ratio']) else 0)
                dev = v / lc - 1
        note = ''
        if typ in ('swap', 'conversion') and np.isfinite(dev) and abs(dev) > 0.05 and '除權息' not in body:
            f = FL[succ].loc[:lt].iloc[-250:]
            if (f.to_numpy() == 1).any():
                note = f'有效比例未核實（文字比例 {o["ratio"]:.6g}，lt 偏離 {dev:+.1%}）'; typ = 'swap_ur'; o['ratio'] = np.nan
        if typ == 'cash': dev = o['cash'] / lc - 1
        rows.append(dict(code=code, type=typ, cash=o['cash'], ratio=o['ratio'], pref_ratio=o['pref_ratio'], pref_value=o['pref_value'], successor=succ,
                         last_trade=str(lt.date()), terminate_date=str(term.date()) if term is not None else '', last_close=lc, impl_ratio=impl, dev_lt=dev,
                         n_evidence=len(ev.get(code, [])), evidence_sha=' | '.join(x['sha'] for x in ev.get(code, [])), note=note))
    T = pd.DataFrame(rows)
    T.to_csv(HERE / 'rp_events.csv', index=False)
    print(T[['code', 'type', 'cash', 'ratio', 'pref_ratio', 'pref_value', 'successor', 'last_trade', 'terminate_date', 'dev_lt', 'note']].to_string())


if __name__ == '__main__':
    main()
