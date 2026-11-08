#!/usr/bin/env python3
"""步驟二：對價表 v2（events_v2.csv）。不看任何策略報酬。
python3 -I -B s3_events.py
(1) 普查到的被持有事件（census.json，U 面板 23 起＋IA 面板價格缺口 6 起）逐筆用公開文字證據核實：
    證據來源＝GitHub 公開 repo 的本地 clone（唯讀、視為不可信文字）。每段證據摘錄寫到 raw/ev_<code>_<k>.txt，
    記摘錄 sha256、原檔 sha256、repo commit、GitHub URL；程式斷言摘錄含關鍵數字字串（數字就是從摘錄抄的，不是記憶）。
    MOPS／證交所／櫃買／web.archive.org 本環境連不到（s0 只測一次，見 env.json），所以沒有官方原頁。
(2) 再擴到 2006～2026 全部國內下市事件（delist_official.csv 四碼 2006 起＋面板 2024-09 後永久停止交易者）：
    非被持有事件只沿用 o3 表的型態並一律標「未核實」。
欄位 verified：已核實（新聞）／已核實（論壇，弱）／類型已核實、對價未核實／未核實。
原表 merger_consideration.csv 的 'knowledge' 一律改寫成上述之一（不沿用研究員記憶）。"""
import sys, json, re, subprocess
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cz_lib as Z

RAW = Z.HERE / 'raw'
CT = 'o_index_deletion_flow/raw/aRthur08701244_Intern-in-CTBC'
CTP = CT + '/AI_Investment/OpenIEStanza_EventEmbedding/input/EB_EventExtraction2/'
PTT = 'o_rev_gap_news_2013/raw/ji08kk44_BDSE26_TEAM3'
PTTP = PTT + '/ptt/ptt_csv(date)/'
NN = 'o_index_deletion_flow/raw/nick_news'
REPO = {CT: 'aRthur08701244/Intern-in-CTBC', PTT: 'ji08kk44/BDSE26_TEAM3', NN: 'nickchengtw/stock-llm-neuralprophet'}

# 證據：code → [(來源檔, 必須逐字出現的句子)]；被採用的數字必須在句子中
EVID = {
    '2847': [(CTP + '201510_6.csv', '每股大眾銀普通股將換發現金8.15元及元大金普通股0.460972股')],
    '3697': [(CTP + '20138_19.csv', '聯發科將以0.794股加 1 元現金，換發 1 股晨星股份'),
             (CTP + '20138_20.csv', '雙方同意擬變更本合併案基準日為2014年 2 月 1 日，原換股比例不變')],
    '2833': [(CTP + '20156_11.csv', '以每股台壽保普通股轉換1.44股中信金普通股')],
    '2823': [(CTP + '202110_9.csv', '開發金本次股份轉換的對價為中國人壽每一股普通'),
             (CTP + '20218_2.csv', '0.80 股普通股、0.73 股特別股及現金 11.5 元'),
             (CTP + '20218_2.csv', '特別股每股發行價格 10 元')],
    '2311': [(CTP + '20165_13.csv', '日月光以每股普通股換發新設控股公司0.5股')],
    '2325': [(CTP + '20182_2.csv', '矽品則為每股普通股換發現金 51.2 元對價')],
    '3474': [(CTP + '201610_3.csv', "['每股收購價', '維持', '30元']")],
    '6286': [(CTP + '20159_15.csv', '將由聯發科以每股195元公開收購立錡股權'),
             (CTP + '20159_15.csv', '首階段每股195元收購35-51%股權')],
    '1704': [(CTP + '20191_14.csv', "['KKR', '支付', '每股56元']")],
    '1902': [(CTP + '20203_4.csv', '擬以每股 20.8 元現金對價')],
    '2448': [(CTP + '20207_4.csv', '晶電以 1 股換發該富采投控普通股 0.5 股、隆達則以 1 股換發富采投控普通股 0.275 股'),
             (CTP + '202012_15.csv', '以晶電最後交易日收盤價 41.4 元計算，換算富采投控開始交易基準價為 82.8 元')],
    '3698': [(CTP + '20207_4.csv', '晶電以 1 股換發該富采投控普通股 0.5 股、隆達則以 1 股換發富采投控普通股 0.275 股')],
    '2394': [(PTTP + '2020-03-28.csv', '普立爾3.9股換1股鴻海')],
    '3009': [(CTP + '20144_3.csv', '以群創為存續公司'), (PTTP + '2019-05-10.csv', '以小吃大，併了奇美集團')],
    '3012': [(CTP + '20199_2.csv', '廣輝併入友達')],
    '3063': [(CTP + '201810_10.csv', '2010 年推動旗下飛信與頎邦合併')],
    '2452': [(CTP + '20188_12.csv', '台達電於 2009 年底宣布公開收購乾坤')],
    '2473': [(PTTP + '2019-05-17.csv', '思源真的是好股票  可惜我買了之後不久就收購下市')],
    '2403': [(CTP + '20172_12.csv', '子公司世平與友尚')],
    '2446': [(CTP + '201512_27.csv', '持股全懋轉給欣興')],
    '2432': [(CTP + '20131_26.csv', 'Gateway、Packard Bell、eMachines與倚天')],
    '2888': [(NN + '/data/news/news_2023-05-17.csv', '台新金併購新光金的意向書也在昨日曝光')],
}
# 依證據定的分類（數字皆來自上面的句子；空白＝證據沒有）。type：
#   cash／conversion／swap／mixed＝條款已核實；swap_ur＝併入上市存續公司已核實、換股比例未核實（用市場隱含比例）；
#   acq_ur＝被併購已核實、對價形式或金額未核實（最後收盤固定、終止日＋20 交易日轉現金）；unknown＝無公開證據。
CLS = {
    '2847': dict(type='mixed', cash=8.15, ratio=0.460972, successor='2885', verified='已核實（新聞）'),
    '3697': dict(type='mixed', cash=1.0, ratio=0.794, successor='2454', base_date='2014-02-01', verified='已核實（新聞）'),
    '2833': dict(type='conversion', ratio=1.44, successor='2891', verified='已核實（新聞）', note='中信金股份轉換台壽保'),
    '2823': dict(type='mixed', cash=11.5, ratio=0.80, pref_ratio=0.73, pref_value=10.0, successor='2883', verified='已核實（新聞）',
                 note='開發金（不是 o3 表的中信金 2891）股份轉換；特別股以發行價 10 元計'),
    '2311': dict(type='conversion', ratio=0.5, successor='3711', verified='已核實（新聞）'),
    '2325': dict(type='cash', cash=51.2, verified='已核實（新聞）', note='原 55 元扣 3.8 元現金股利後 51.2 元'),
    '3474': dict(type='cash', cash=30.0, verified='已核實（新聞）'),
    '6286': dict(type='cash', cash=195.0, verified='已核實（新聞）', note='兩階段皆 195 元'),
    '1704': dict(type='cash', cash=56.0, verified='已核實（新聞）'),
    '1902': dict(type='cash', cash=20.8, verified='已核實（新聞）'),
    '2448': dict(type='conversion', ratio=0.5, successor='3714', verified='已核實（新聞）'),
    '3698': dict(type='conversion', ratio=0.275, successor='3714', verified='已核實（新聞）', note='2020-06 新聞為「暫訂」比例'),
    '2394': dict(type='swap', ratio=1 / 3.9, successor='2317', verified='已核實（論壇，弱）', note='PTT 推文「3.9 股換 1 股鴻海」'),
    '3009': dict(type='swap_ur', successor='3481', verified='類型已核實、對價未核實'),
    '3012': dict(type='swap_ur', successor='2409', verified='類型已核實、對價未核實'),
    '3063': dict(type='acq_ur', successor='6147', verified='類型已核實、對價未核實', note='存續頎邦為上櫃股，紙上與 V7 面板都沒有其價格'),
    '2452': dict(type='acq_ur', successor='2308', verified='類型已核實、對價未核實', note='台達電公開收購（現金），金額未核實'),
    '2473': dict(type='acq_ur', verified='類型已核實、對價未核實', note='只有論壇「收購下市」'),
    '2403': dict(type='acq_ur', successor='3702', verified='類型已核實、對價未核實', note='只知成為大聯大子公司'),
    '2446': dict(type='swap_ur', successor='3037', verified='類型已核實、對價未核實'),
    '2432': dict(type='acq_ur', successor='2353', verified='類型已核實、對價未核實', note='只知宏碁擁有倚天品牌；代碼後由倚天酷碁重用'),
    '2888': dict(type='acq_ur', successor='2887', verified='類型已核實、對價未核實', note='只有 2023 意向書 0.584 股（非最終條款）'),
}
DIV_ADJ_TEXT = {'2823'}   # 證據文字明寫「對價以雙方調整除權息後的價格為基礎」
UNK = ['1520', '1716', '6004', '5305', '2456', '2809']   # 普查被持有、但公開來源找不到任何文字證據


def git_head(rel):
    for k in REPO:
        if rel.startswith(k):
            h = subprocess.run(['git', '-C', str(Z.AG / k), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
            return k, h
    return None, None


def main():
    census = json.loads((Z.HERE / 'census.json').read_text(encoding='utf-8'))
    heldU = {e[0]: e[1] for e in census['distinct_perm_events_U']}
    heldI = {e[0]: e[1] for e in census['distinct_perm_events_IA']}
    O3T = pd.read_csv(Z.O3 / 'merger_consideration.csv', dtype={'code': str, 'successor': str}).set_index('code')
    DL = Z.DL.copy()
    # ---- 證據摘錄 ----
    ev_rows = {}
    for code, L in EVID.items():
        for k, (rel, sent) in enumerate(L):
            src = Z.AG / rel
            s = src.read_text(encoding='utf-8', errors='ignore')
            i = s.find(sent); assert i >= 0, (code, rel, sent)
            exc = s[max(0, i - 400): i + len(sent) + 400]
            f = RAW / f'ev_{code}_{k}.txt'
            repo, head = git_head(rel)
            url = f'https://github.com/{REPO[repo]}/blob/{head}/' + rel[len(repo) + 1:]
            hdr = f'# 證據摘錄（不可信文字，只供核對）\n# source_url: {url}\n# source_sha256: {Z.sha(src)}\n# 關鍵句: {sent}\n\n'
            f.write_text(hdr + exc, encoding='utf-8')
            ev_rows.setdefault(code, []).append(dict(raw=f.name, raw_sha256=Z.sha(f), url=url, src_sha256=Z.sha(src), key=sent))
    # 數字必須出現在關鍵句
    def num_in(code, x):
        if x is None: return True
        txt = ' '.join(e['key'] for e in ev_rows.get(code, []))
        cand = {f'{x:g}', f'{x:.2f}', f'{x:.1f}', f'{x:.3f}', str(x), f'{x:.6f}'}
        if code == '2394': cand |= {'3.9'}
        return any(c in txt for c in cand)
    for code, c in CLS.items():
        for k in ('cash', 'ratio', 'pref_ratio', 'pref_value'):
            assert num_in(code, c.get(k)), (code, k, c.get(k))
    # ---- 事件表 ----
    def first_ge(d):
        return int(np.searchsorted(Z.DATES, np.datetime64(pd.Timestamp(d))))
    rows = []
    codes = set(heldU) | set(heldI)
    allc = set(DL4c for DL4c in Z.DL4[Z.DL4.date >= '2006-01-01'].code)
    # 面板 2024-09 之後永久停止交易者（官方名單只到 2024-09-02）
    for j, c in enumerate(Z.COLS):
        w = np.flatnonzero(Z.TRU[:, j])
        if len(w) and Z.IDX[w[-1]] >= pd.Timestamp('2024-09-03') and Z.IDX[w[-1]] < Z.IDX[-1] - pd.Timedelta(days=30):
            allc.add(c)
    for c in sorted(codes | allc):
        j = Z.CI.get(c)
        dd = DL4dates = sorted(Z.DLD.get(c, []))
        term = [d for d in dd if d >= pd.Timestamp('2006-01-01')]
        term = term[0] if term else None
        if j is not None:
            if c in heldU: lt = Z.idx_of(heldU[c])
            else:
                w = np.flatnonzero(Z.TRU[:, j]); w = w[Z.IDX[w] <= term] if term is not None else w
                lt = int(w[-1]) if len(w) else -1
        else:
            lt = -1
        name = Z.DL4.set_index('code')['name'].get(c, O3T['name'].get(c, '') if c in O3T.index else '')
        base = dict(code=c, name=name, held=c in codes, held_U=c in heldU, held_IA_only=(c in heldI and c not in heldU),
                    last_trade=str(Z.IDX[lt].date()) if lt >= 0 else '', last_trade_ia=heldI.get(c, ''),
                    terminate_date=str(term.date()) if term is not None else '', last_close=float(Z.CU[lt, j]) if lt >= 0 else np.nan)
        if c in CLS:
            r = dict(type='', cash=np.nan, ratio=np.nan, pref_ratio=np.nan, pref_value=np.nan, successor='', base_date='', note='')
            r.update(CLS[c]); base.update(r)
        elif c in UNK and c in codes:
            base.update(type='unknown', cash=np.nan, ratio=np.nan, pref_ratio=np.nan, pref_value=np.nan, successor='', base_date='',
                        verified='未核實', note='被持有；公開來源（可達者）找不到文字證據' + (f"；o3 表記為 {O3T.loc[c, 'type']}（研究員記憶）" if c in O3T.index else ''))
        else:
            t0 = O3T.loc[c, 'type'] if c in O3T.index else 'unknown'
            base.update(type='unknown', cash=np.nan, ratio=np.nan, pref_ratio=np.nan, pref_value=np.nan, successor='', base_date='',
                        verified='未核實', note=(f'o3 表記為 {t0}（研究員記憶，未核實）' if c in O3T.index else '無資料'))
        if not base['terminate_date'] and lt >= 0:
            base['terminate_date'] = ''   # 官方名單沒有（2024-09 之後）：引擎以最後交易日＋14 天代替
        s = base.get('successor') or ''
        sj = Z.CI.get(s) if s else None
        tdate = pd.Timestamp(base['terminate_date']) if base['terminate_date'] else (Z.IDX[lt] + pd.Timedelta(days=14) if lt >= 0 else None)
        if base.get('base_date'): tdate_s = max(pd.Timestamp(base['base_date']), tdate) if tdate is not None else pd.Timestamp(base['base_date'])
        else: tdate_s = tdate
        sfirst = ''
        if sj is not None and tdate_s is not None:
            w = np.flatnonzero(Z.TRU[:, sj]); w = w[w >= first_ge(tdate_s)]
            sfirst = str(Z.IDX[w[0]].date()) if len(w) else ''
        base['succ_first_trade'] = sfirst
        base['succ_in_panel'] = sj is not None
        base['pay_date'] = ''
        # 隱含比例與偏差（只用事件資料，不用策略報酬）
        impl = np.nan; dev_lt = np.nan; dev_s = np.nan
        if sj is not None and lt >= 0:
            if np.isfinite(Z.CU[lt, sj]): impl = base['last_close'] / Z.CU[lt, sj]
            def val_at(t):
                v = (base['ratio'] if np.isfinite(base.get('ratio', np.nan)) else 0) * Z.CU[t, sj]
                v += base['cash'] if np.isfinite(base.get('cash', np.nan)) else 0
                v += (base['pref_ratio'] * base['pref_value']) if np.isfinite(base.get('pref_ratio', np.nan)) else 0
                return v
            if base['type'] in ('swap', 'conversion', 'mixed'):
                if np.isfinite(Z.CU[lt, sj]): dev_lt = val_at(lt) / base['last_close'] - 1
                if sfirst: dev_s = val_at(Z.idx_of(sfirst)) / base['last_close'] - 1
        if base['type'] == 'cash' and lt >= 0: dev_lt = base['cash'] / base['last_close'] - 1
        base.update(impl_ratio=impl, dev_vs_lastclose_lt=dev_lt, dev_vs_lastclose_s=dev_s)
        ev = ev_rows.get(c, [])
        base['source_url'] = ' | '.join(e['url'] for e in ev)
        base['raw_files'] = ' | '.join(e['raw'] for e in ev)
        base['raw_sha256'] = ' | '.join(e['raw_sha256'] for e in ev)
        base['o3_type'] = O3T.loc[c, 'type'] if c in O3T.index else ''
        base['o3_verified'] = O3T.loc[c, 'verified'] if c in O3T.index else ''
        rows.append(base)
    T = pd.DataFrame(rows)
    cols = ['code', 'name', 'type', 'verified', 'held', 'held_U', 'held_IA_only', 'announce_date', 'base_date', 'last_trade', 'last_trade_ia',
            'terminate_date', 'last_close', 'cash', 'ratio', 'pref_ratio', 'pref_value', 'successor', 'succ_in_panel', 'succ_first_trade',
            'pay_date', 'impl_ratio', 'dev_vs_lastclose_lt', 'dev_vs_lastclose_s', 'source_url', 'raw_files', 'raw_sha256', 'o3_type', 'o3_verified', 'note']
    T['announce_date'] = ''
    # ---- 迭代 v2／v3（資料核實，不看任何報酬）：條款已核實的純換股／股份轉換，若 lt 當日市場隱含價值偏離 > 5%，
    #      且存續公司在 lt 前 250 個交易日內有除權息（flag 1），而證據文字沒有說明條款已調整除權息 → 有效比例未核實，改 swap_ur
    FL = pd.read_parquet(Z.PANEL / 'flag.parquet')
    recl = []
    for i, r in T.iterrows():
        if r.type not in ('swap', 'conversion') or not np.isfinite(r.dev_vs_lastclose_lt) or abs(r.dev_vs_lastclose_lt) <= 0.05: continue
        if r.code in DIV_ADJ_TEXT: continue
        lt = pd.Timestamp(r.last_trade); f = FL[r.successor].loc[:lt].iloc[-250:]
        exd = [str(d.date()) for d in f.index[f.to_numpy() == 1]]
        if exd:
            recl.append(dict(code=r.code, from_type=r.type, dev_lt=float(r.dev_vs_lastclose_lt), succ_exdiv=exd, text_ratio=float(r.ratio)))
            T.loc[i, 'type'] = 'swap_ur'; T.loc[i, 'verified'] = '類型已核實、有效比例未核實'
            T.loc[i, 'note'] = (r.note + '；' if r.note else '') + f'文字比例 {r.ratio:.6g} 早於存續公司除權（{exd[-1]}），lt 當日隱含比例 {r.impl_ratio:.4f}，偏離 {r.dev_vs_lastclose_lt:+.1%}'
            T.loc[i, 'ratio'] = np.nan
    T = T[cols]
    T.to_csv(Z.HERE / 'events_v2.csv', index=False)
    H = T[T.held]
    V = H[H.verified == '已核實（新聞）']
    devs = [abs(x) for x in V.dev_vs_lastclose_lt if np.isfinite(x)]
    d_max = float(max(devs))
    summ0 = dict(d_max_verified_same_day=d_max, d_used=float(np.ceil(d_max * 1000) / 1000), verified_dev_lt={r.code: float(r.dev_vs_lastclose_lt) for r in V.itertuples() if np.isfinite(r.dev_vs_lastclose_lt)})
    summ = dict(reclassified_v2_v3=recl, n_rows=len(T), n_held=int(T.held.sum()), n_held_U=int(T.held_U.sum()), n_held_IA_only=int(T.held_IA_only.sum()),
                held_by_type=H.type.value_counts().to_dict(), held_by_verified=H.verified.value_counts().to_dict(),
                all_by_verified=T.verified.value_counts().to_dict(), sha256=Z.sha(Z.HERE / 'events_v2.csv'),
                d=summ0, o3_corrections=[dict(code=r.code, o3=r.o3_type, v2=r.type, note=r.note) for r in H.itertuples() if r.o3_type and r.o3_type != r.type])
    (Z.HERE / 'events_v2_summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, default=str)[:3000])
    print(H[['code', 'type', 'verified', 'last_trade', 'terminate_date', 'last_close', 'cash', 'ratio', 'successor', 'succ_first_trade', 'impl_ratio', 'dev_vs_lastclose_lt', 'dev_vs_lastclose_s']].to_string())


if __name__ == '__main__':
    main()
