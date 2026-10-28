#!/usr/bin/env python3
"""凍結：依 PLAN 的選版規則（只用挑選期 iterations.json）選定版本，寫 PREREG（sha256＋時間戳）到 results.json。之後才能跑 oos.py。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE / 'results.json'
res = json.loads(RES.read_text(encoding='utf-8'))
assert 'PLAN_sha256' in res and 'PREREG' not in res, 'PREREG 已存在'
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
vs = it['versions']
assert len(vs) <= 12
ok = {k: v for k, v in vs.items() if v['normal']['paired_vs_J1']['mean_pp'] >= -0.05}
nk = lambda k: len(vs[k]['spec']['js'])
sel = min(ok, key=lambda k: (round(ok[k]['normal']['ex_sd_pp'], 6), nk(k)))
s = vs[sel]['spec']
PREREG = {
 'selected_version': sel, 'spec': s,
 'spec_text': f"{len(s['js'])} 批，批 i 在「營收期限 D_m 之後第 j_i 個交易日」（j＝{s['js']}）收盤算訊號、隔日開盤換成該批前 10 名；"
              f"{'a0：其他批持有的股票照樣重複買' if s['a']=='a0' else 'a1：跳過他批持股'}；{'b0：每批完整重排（AB）' if s['b']=='b0' else 'b1：j>1 批營收腿凍結在 J+1 橫斷面'}；"
              f"{'c0：批間不再平衡' if s['c']=='c0' else 'c1：每月官方換股日成交日開盤把各批調回等額（只對淨買賣計成本）'}。"
              '其餘＝主策略 V3：52 週高點百分位 0.5＋近 3 月營收年增率百分位 0.5、前 300 流動性池（60 日均成交值 ≥1,000 萬、股價 ≥10）、排除金融、官方處置名單 E1 排除、'
              '開盤漲停且該批未持有→遞補、停牌凍結、X 併購出清、P 下市股價格、R_DEL 營收（2021-03 起 rev_5y）、AB 只用已公告營收。',
 'selection_trace': {'passed_B1_filter': sorted(ok), 'ex_sd_pp': {k: vs[k]['normal']['ex_sd_pp'] for k in vs}, 'paired_mean_pp': {k: vs[k]['normal']['paired_vs_J1']['mean_pp'] for k in vs},
                     'edge_note': '選中版本與鄰居 V03（同位置、c0）只差 c 選項；V03 的配對差 −0.0507 剛好落在 −0.05 門檻外，V11（再疊 b1）+0.015。屬門檻邊緣，報告時標註贏家詛咒。'},
 'n_versions_tested': len(vs), 'n_J_scan': 21,
 'multiple_testing': {'m_main_history': 10, 'm_this_topic': 21 + len(vs), 'm_total': 10 + 21 + len(vs), 'rule': 'Bonferroni 單尾 p×m'},
 'oos_runs': ['全期 21 個 J 版本（一般成本、AB seed 0）：自己的窗 t、日曆月 t、S1～S4、VAL、HOLD、最大回撤；J0 另報 seed 0～4；分布與官方日百分位；判定 A',
              '官方日曆：V3 正式（OFF，未做 AB）與官方 AB（seed 0～4）全期',
              '全部 12 個分批版本的 VAL、HOLD、POST、ALL（一般、壓力 flat、壓力 +0.3%、扣成本前）：對 J+1、OFF、0050、同池等權',
              '選中版本：端點敏感度、損益兩平滑價（格點 0、0.1、0.2、0.3、0.5、0.8、1.2% 線性內插）、1,000 萬容量、Bonferroni、分年、分產業貢獻、去掉最大貢獻股',
              '描述性（事前加列）：分批組合在「整體平移位置」下的 t 分布（T4 間距 5：j0＝−8…−3；T2 間距 10：j0＝−8…+2），與單日版 t 分布比較寬度',
              'V4 尚未凍結：補報欄位留空，待 o_v4_revenue_patch_official 凍結後重跑 oos.py 的 J 掃描'],
 'final_rules': res['PLAN']['final'],
 'periods': res['PLAN']['periods'],
}
txt = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
res['PREREG'] = PREREG
res['PREREG_sha256'] = hashlib.sha256(txt.encode('utf-8')).hexdigest()
res['PREREG_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
res['ITERATIONS_sha256'] = hashlib.sha256((HERE / 'iterations.json').read_bytes()).hexdigest()
res['EXPLORE_sha256'] = hashlib.sha256((HERE / 'explore.json').read_bytes()).hexdigest()
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(sel, res['PREREG_sha256'], res['PREREG_time_utc'])
