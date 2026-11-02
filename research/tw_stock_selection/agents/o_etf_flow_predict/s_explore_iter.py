"""彙整 explore.json（只含不看報酬的 P1、劑量描述、MDE，以及臺灣50 低劑量對照的描述值）與 iterations.json（P1 重建器的全部版本）。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eflib as E
H = E.HERE
p1t = json.load(open(H / 'p1_tw50.json')); p10 = json.load(open(H / 'p1_0056.json'))
dm = json.load(open(H / 'dose_overlap_mde.json')); ctl = json.load(open(H / 'tw50_control.json')); gate = json.load(open(H / 'gate.json'))
explore = {
 'gate': {k: {kk: v[kk] for kk in ('reviews_expected', 'reviews_complete', 'list_completeness', 'aum_monthly_coverage', 'pass_list', 'pass_aum')} for k, v in gate['etf'].items()},
 'gate_decision': gate['decision'],
 'news_archive_0056_lists': '2013-03（5 入 5 出，完整）、2015-03（只見裕日車納入）、2018-12（5 入，刪除只列 2/5）；其餘 2007～2022 無',
 'P1_tw50': {k: {'add': v['add'], 'del': v['dele']} for k, v in p1t.items()},
 'P1_0056': {k: {kk: v[kk] for kk in ('n_add', 'add_recall_top35', 'add_recall_top50', 'add_rank_median', 'n_del', 'del_recall_rank66_or_out')} for k, v in p10.items()},
 'D_0056_adds_desc': dm['D_0056_adds'], 'D_tw50': dm['D_tw50_pred_adds_per_1000億AUM'], 'overlap_main': dm['overlap_with_main_B10_seed0'], 'P4_MDE': dm['P4_MDE'],
 'tw50_low_dose_control（描述，2019 後已被看過）': {k: v for k, v in ctl.items() if '|all|' in k or '|sel' in k},
}
json.dump(explore, open(H / 'explore.json', 'w'), ensure_ascii=False, indent=1)
its = [
 {'id': 'T1', 'etf': 'TW50', 'spec': 'v1 PIT 股數當日', 'objective': 'P1（不看報酬）', 'result': p1t['v1_pit']['add'] | {'del': p1t['v1_pit']['dele']}},
 {'id': 'T2', 'etf': 'TW50', 'spec': 'v2 PIT 股數落後 63 日', 'objective': 'P1', 'result': p1t['v2_pit_lag63']['add'] | {'del': p1t['v2_pit_lag63']['dele']}},
 {'id': 'T3', 'etf': 'TW50', 'spec': 'v3 v1＋60 日均值前 300 流動性', 'objective': 'P1', 'result': p1t['v3_pit_liq300']['add'] | {'del': p1t['v3_pit_liq300']['dele']}},
 {'id': 'T4', 'etf': 'TW50', 'spec': 'v4 v1＋修正 4763 股數（凍結）', 'objective': 'P1', 'result': p1t['v4_pit_fix4763']['add'] | {'del': p1t['v4_pit_fix4763']['dele']}},
 {'id': 'T5', 'etf': 'TW50', 'spec': 'v5 v4＋落後 63 日', 'objective': 'P1', 'result': p1t['v5_pit_fix4763_lag63']['add'] | {'del': p1t['v5_pit_fix4763_lag63']['dele']}},
] + [{'id': f'H{i+1}', 'etf': '0056', 'spec': k, 'objective': 'P1 recall（precision 不可算）', 'result': {kk: p10[k][kk] for kk in ('add_recall_top35', 'add_recall_top50', 'add_rank_median', 'del_recall_rank66_or_out')}}
     for i, k in enumerate(['y1', 'y2', 'y3', 'y4', 'y5'])] + [
 {'id': 'C1', 'etf': 'TW50', 'spec': '低劑量對照：預測納入 W1/W2/W3 × 預測刪除 W1/W2/W3（6 格，描述性，未選版）', 'objective': '對照（etf_rebalance 一致性）',
  'result': {k: ctl[k] for k in ('add|all|W1|net_ex', 'add|all|W2|net_ex', 'add|all|W3|net_ex', 'del|all|W1|net_ex', 'del|all|W2|net_ex', 'del|all|W3|net_ex')}}]
json.dump({'n_versions': len(its), 'note': 'P1 版本只以名單命中率為目標、不看報酬；C1 為唯一碰報酬的描述格（6 格），沒有選版。SPA 族群新增 16 條（10 個 P1 版本不計入報酬族群，6 格 C1 計入）。', 'versions': its},
          open(H / 'iterations.json', 'w'), ensure_ascii=False, indent=1)
print('ok', len(its))
