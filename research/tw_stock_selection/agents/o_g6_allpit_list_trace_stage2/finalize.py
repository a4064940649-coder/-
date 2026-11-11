# -*- coding: utf-8 -*-
"""收尾：斷言 PREREG 存在 → 彙整 explore.json、iterations.json、results.json 的 RESULTS（只讀 out/ 的結果檔）。python3 -I -B finalize.py"""
import json, hashlib, os, datetime
HERE = os.path.dirname(os.path.abspath(__file__)); O = HERE + '/out'
R = json.load(open(HERE + '/results.json'))
assert 'PREREG' in R and hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
L = lambda f: json.load(open(f'{O}/{f}'))
b3, b9, p2, c = L('b3_perj.json'), L('b9_align.json'), L('p2_paper_compare.json'), L('c_review.json')
au, aup, ac = L('a_U_2006_2013.json'), L('a_U_2006_2013_posthoc.json'), L('a_compare_u.json')
AIT = json.load(open(HERE + '/a_iterations.json'))
explore = dict(
  note='探索（B 引擎層與 A 名單層的描述統計；不產生任何新選股訊號）',
  nav_first_divergence='b1_navdiff.json：複製員對原引擎（orig 名單）逐 j 種子平均淨值，100 萬、300 萬、1,000 萬、壓力四格全部在 2006-09～2007-12 就出現 >1e−9 的日報酬差（最早 2006-09-22／2006-11-22，20 個種子同值），第一天差 0.01～0.25bp；不是 1,000 萬獨有。',
  input_data_diff='兩引擎輸入 C、O、V、CF、PC、ADJ、G 逐格完全相同；SIG（t−1 位移後）相同；V60 有 6,389 格 NaN 不一致；vrank（成交值名次）有 306,097／2,837,461 格（10.8%）不同、最大差 487 名。',
  material_divergence_cases={'N100 j6 2007-04-18': '3007 綠點（表外永久停牌，delist_official 終止日 2007-04-13、最後成交 2007-04-12、不在併購表）：複製員在 04-18 換股日轉現金，原引擎等到最後成交＋14 曆日（04-26）之後的 05-21 換股日才轉；部位約 7.8% 淨值',
                             'N100／S100 j6 2013-10-22': '3057 喬鼎 2013-09-02～09-30 停牌：09-18 換股日停牌的非名單持股，複製員下順延賣單（10-01 恢復後每日 1 張，V60 缺值），原引擎不下單、等 10-22 換股日；3057 在 10 月漲 25.35→33.15，差約 3.7 萬元（約 2% 淨值）',
                             'N1000 j8 2018-03-23': '6414：上游 32 股差（4013 對 3981）跨過整股邊界；2020-10 前盤後零股上限 74 股綁住 → 複製員剩 907 股（約 4.3% 淨值）賣不掉、現金不足使 2915 整筆買單順延（F3 整筆成本判定）'},
  yearly_2026=b3)
json.dump(explore, open(HERE + '/explore.json', 'w'), ensure_ascii=False, indent=1, default=float)
iters = dict(
  n_versions_total='A：C0 閘門 1 版（v1 一次過閘）＋U 嚴格版 1 次＋事後名稱來源修正 1 版；B：對齊實驗 V0～V3 四版 × 3 格（N1000、S100、O30）；K：紙上三遮罩 M0／M1／M2',
  A=[dict(version=x['version'], jaccard_C0=x['jaccard_mean'], slots_diff=x['slots_diff']) for x in AIT],
  A_U_strict={k: dict(n_2006_2012=v['n_2006_2012'], n_2006_2013=v['n_2006_2013']) for k, v in au.items()},
  A_U_posthoc={k: dict(n_2006_2012=v['n_2006_2012'], n_2006_2013=v['n_2006_2013']) for k, v in aup.items()},
  A_compare_strict=dict(official=ac['official_n'], mine=ac['mine_n'], only_mine=len(ac['only_mine']), only_off=len(ac['only_off'])),
  B_variants={'V0': '複製員引擎原樣（重現 rep_o19 cache）', 'V1': 'V0＋資料對齊（vrank／V60 換成原引擎 VRp／V60p）', 'V2': 'V1＋E23 表外永久停牌終止日改為最後成交＋14 曆日（原引擎讀法）',
              'V3': 'V2＋換股日停牌的非名單持股不下順延賣單（原引擎讀法）'},
  B_results=b9, K_masks={'M0': 'o18 原管線', 'M1': '審查員修正（M0 ∪ 2021-09-23 後公告≤d 且涵蓋 e）', 'M2': '全期統一：只用官方表「公告 ≤ d 且涵蓋 e」'})
json.dump(iters, open(HERE + '/iterations.json', 'w'), ensure_ascii=False, indent=1, default=float)
R['RESULTS'] = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), perj_V0=b3, align=b9, paper=p2['stats'], paper_list_diffs={k: {kk: vv for kk, vv in v.items() if kk != 'days'} for k, v in p2['list_diffs'].items()},
                    c_review=c, A=iters['A'], A_U_strict=iters['A_U_strict'], A_U_posthoc=iters['A_U_posthoc'], A_compare_strict=iters['A_compare_strict'])
json.dump(R, open(HERE + '/results.json', 'w'), ensure_ascii=False, indent=1, default=float)
print('ok')
