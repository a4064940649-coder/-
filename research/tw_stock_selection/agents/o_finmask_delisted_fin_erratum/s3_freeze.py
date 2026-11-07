#!/usr/bin/env python3
"""第 3 步：凍結 ERRATUM_SPEC_FM.json（含 sha256 與 UTC），並在 results.json 寫入 PREREG。之後才可計數。
用法：python3 -I -B s3_freeze.py"""
import sys, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freg as R
DES, DESREG = R.check_reg('PREREG_DESIGN')
import fclass as F
import flib as L
import numpy as np, pandas as pd
AG = R.AG
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
fin = IT['versions'][-1]
KW = fin['classifier']['kw']; EXCL = fin['classifier']['excl_f2']; NEX = fin['classifier']['name_excl']
CL = F.Classifier(KW, EXCL, NEX)
days = [str(d.date()) for d in L.VM(lambda d: L.C0MASK).sig_days if L.E.LO <= d <= L.E.HI]
# 分類結果指紋（全部 1,249 檔 × 93 日；不是名單）
rows = []
for d in days:
    for j, c in enumerate(L.COLS):
        s, _ = CL.f1(c, d); c0 = bool(L.C0MASK[j])
        v1 = CL.f1_value(s, c0); v2 = CL.f2(c)
        if v1 != c0 or v2 != c0: rows.append([d, c, int(c0), int(v1), int(v2), s])
fp = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()
ER2S = json.loads((AG / 'o_rev_trunc_2009_erratum/ERRATUM2_SPEC.json').read_text(encoding='utf-8'))
srcs = {('o13/' + k): R.sha(F.O13 / k) for k in list(F.SNAP_FILES) + list(F.LATER_FILES)}
srcs.update({'stock_list.csv(2026)': R.sha(F.SL_PATH), 'tw_revenue.csv': R.sha(F.REV_PATH), 'listing_master.csv(V7)': R.sha(F.LM_PATH),
             'delist_official.csv': R.sha(F.DELIST_PATH), 'industry_pit_o12.csv(不採用，只記錄)': R.sha(AG / 'o_pit_shares_industry_2006_2013/raw/center/industry_pit_o12.csv')})
SPEC = dict(
    utc=R.now_utc(),
    title=DES['title'], prereg_design=dict(file='PREREG_DESIGN.json', sha256=DESREG['sha256'], utc=DESREG['utc']),
    no_returns=DES['no_returns'],
    scope='主策略 2006-01～2013-12 名單；只計數。不產生新選股訊號，不重算報酬（符合 S1）',
    pipeline=dict(
        engine="main_v2/v2_core.V2('gapfix', YL, Y5, exfin=True, unavail=None)；plan(10, 'official')（同分：分數高→y3 高→欄位順序小；開盤漲停且上期未持有→遞補）",
        signal_days='V2.sig_days 落在 2006-01-01～2013-12-31 者：93 日（2006-04-11～2013-12-11）；2006～2012 為 81 日',
        pool='top300(d)＝60 日均成交值 V60 ≥ 1,000 萬且收盤 ≥ 10 的股票，依 V60 由大到小（stable）取前 300',
        eligible='el＝top300 ∩ y3 有值（且非遮罩） ∩ 52 週分數有值 ∩ 非處置 ∩ 隔日有開盤成交',
        mask_injection=DES['base']['mask_injection'],
        U_opt=dict(affected_set=ER2S['affected_set'], U_opt=ER2S['U_opt'], note='遮罩替換後，受影響集合 A 的「非金融」條件也改用 mask_F(d)'),
        validity=DES['base']['pipeline_validity'], sample_days=['2006-04-11', '2008-04-11', '2009-11-11', '2011-07-11', '2013-12-11']),
    classifiers=dict(
        F1=DES['classifiers']['F1'], F2=DES['classifiers']['F2'], F3=DES['classifiers']['F3'], UNKNOWN=DES['classifiers']['UNKNOWN'], mask=DES['classifiers']['mask'],
        final_version=fin['version'], keywords=KW, f2_manual_exclusion=EXCL, name_exclusion=NEX,
        snapshot_effective_dates=F.SNAP_FILES, later_sources_dates=F.LATER_FILES, fin_labels=list(F.FIN_LABELS), tse_fin_code=F.TSE_FIN,
        name_sources='stock_list 2026、o13 快照、center StockList 2015／2016／2017、ISIN 2023、tw_revenue、delist_official 的名稱聯集',
        delisted_def='已下市＝不在 2026 stock_list，或 listing_master 下市日不空白',
        fingerprint=dict(desc='對 1,249 檔 × 93 日，列出 F1 或 F2 遮罩值與 C0 不同的 [訊號日, 代號, C0, F1, F2, F1 狀態]（依訊號日、欄位順序）後 json.dumps(ensure_ascii=False) 的 sha256',
                         n_rows=len(rows), sha256=fp,
                         codes_F1_diff=sorted({r[1] for r in rows if r[2] != r[3]}), codes_F2_diff=sorted({r[1] for r in rows if r[2] != r[4]}))),
    units=DES['units'], judgement=DES['judgement'],
    descriptive=dict(
        F3='F3 遮罩下同樣 4 個名單組合 × 2 分母',
        ex_top='對每個 (名單, 分類器)：遮罩差異代號逐一單獨套用（其他代號維持 C0），異動槽位最多者＝最大貢獻股（同數取代碼小者）；把它還原成 C0 後重跑，報比例',
        decomposition='(i) 只加入「C0 未遮、F 判金融、已下市」的代號；(ii) 只改其他遮罩差異代號（現存股改判，例如 2841、5871）；各自重跑報異動',
        by_year='總異動依訊號日年份',
        unknown='UNKNOWN_EVID／UNKNOWN_NOEVID 代號數、池內股月、出現在名單的槽位',
        direct='直接污染：C0 未遮、F 判金融、已下市的代號在 C0、U_opt 名單（C0 遮罩）中的槽位；另報 F1＝UNKNOWN 但 F2＝金融者',
        executability='直接污染槽位的進場日、開盤漲停、持有期內是否停止交易、距最後交易日的交易日數（只描述）',
        P4_measure=DES['predictions']['P4']),
    predictions=DES['predictions'],
    errata=dict(
        E1=DES['errata']['E1'] + '。實作：erlib.SURV 只留 TRUNC；不含 5854＝5854 不放回（原因改 DROPPED）；另報不含 5854 與 2473；比例＝放回股進榜槽位 ÷ 810／930',
        E2=DES['errata']['E2'] + '。改標規則：受影響股若 listing_master 下市日空白且在 2026 stock_list（仍上市），其任何 SURV 原因都改為 LISTED_GAP（不放回）；以 drop＝這些代號重跑 U_opt',
        E3=DES['errata']['E3'], E4=DES['errata']['E4'],
        E5=DES['errata']['E5'],
        E6=DES['errata']['E6'] + '。實作同 o15 審查：28／58／60 開頭代號原因改 DROPPED 後重跑 U_opt（C0 遮罩）；另報 F2 遮罩下 U_opt 放回槽位（相對 C0_F2 基準）',
        format=DES['errata']['format']),
    files=dict(code={k: R.sha(HERE / k) for k in ('freg.py', 'fclass.py', 'flib.py', 's0_prereg_design.py', 's1_explore.py', 's2_iterate.py', 's3_freeze.py')},
               engine=dict(v2_core=R.sha(AG / 'main_v2/v2_core.py'), ia_core=R.sha(AG / 'independent_audit/ia_core.py'), erlib=R.sha(AG / 'o_rev_trunc_2009_erratum/erlib.py')),
               refs=dict(ERRATUM2_json=R.sha(AG / 'o_rev_trunc_2009_erratum/ERRATUM2.json'), ERRATUM2_SPEC=R.sha(AG / 'o_rev_trunc_2009_erratum/ERRATUM2_SPEC.json'),
                         o14_ERRATUM_json=R.sha(AG / 'o_g8_protocol_fin_erratum/ERRATUM.json'), explore_json=R.sha(HERE / 'explore.json'), iterations_json=R.sha(HERE / 'iterations.json')),
               sources=srcs),
    replication='複製員只讀本檔：自行從原始檔重建 F1／F2（比對 fingerprint），自行寫遮罩替換、名單比對與計數；可 import v2_core／ia_core 的面板與滾動訊號原語；cache 放 scratchpad，結束時刪除。8 個比例與異動槽位清單要逐筆相同。',
)
h = R.register('ERRATUM_SPEC_FM', SPEC, 'ERRATUM_SPEC_FM.json')
r = R.load_res(); r['PREREG'] = dict(file='ERRATUM_SPEC_FM.json', sha256=h, utc=SPEC['utc'], note='凍結後才可執行 s4_count.py'); R.save_res(r)
print('ERRATUM_SPEC_FM', h, SPEC['utc'], len(rows), fp)
