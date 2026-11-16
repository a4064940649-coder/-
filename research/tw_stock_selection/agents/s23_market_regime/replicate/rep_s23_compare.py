#!/usr/bin/env python3
"""逐數比對：複製 results_rep.json 對原研究 results_train.json／results.json／results_posthoc.json（只讀原研究輸出的 JSON 數字）。
結果寫回 results_rep.json 的 comparison 欄。python3 -I -B rep_s23_compare.py"""
import json, math
from pathlib import Path

H = Path(__file__).resolve().parent
O = H.parent
R = json.loads((H / 'results_rep.json').read_text())
OT = json.loads((O / 'results_train.json').read_text())
OV = json.loads((O / 'results.json').read_text())
OP = json.loads((O / 'results_posthoc.json').read_text())
FZ = R['train']['frozen_rep']


def flat(x, p=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f'{p}/{k}')
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield p, float(x)


def cmp(a, b):
    fa = dict(flat(a)); fb = dict(flat(b))
    common = [k for k in fa if k in fb and math.isfinite(fa[k]) and math.isfinite(fb[k])]
    d = [abs(fa[k] - fb[k]) for k in common]
    return dict(n_compared=len(common), max_abs_diff=max(d) if d else None, only_rep=len(set(fa) - set(fb)), only_orig=len(set(fb) - set(fa)))


out = {}
# 訓練期 8 設定 × 20 路徑（train、train_dd、calmar、y1618、y1921、seg 2018／2020-03）
tr = {}
for c, D in OT['paths'].items():
    mine = {k: {f: R['paths_cont'][c][k][f] for f in ('train', 'train_dd', 'calmar', 'y1618', 'y1921')} | {'seg': {g: R['paths_cont'][c][k]['seg'][g] for g in ('2018', '2020-03')}} for k in D}
    tr[c] = cmp(mine, D)
out['train_paths'] = tr
s = {}
for c in R['train']['summary']:
    if c == 'CUR':
        continue
    a = R['train']['summary'][c]; b = OT['summary'][c]
    s[c] = dict(n_improved=(a['n_improved'], b['n_improved']), cagr_diff_median_pp=(a['cagr_diff']['median_pp'], b['cagr_diff_median_pp']),
                dd_diff_median_pp=(a['dd_diff']['median_pp'], b['dd_diff_median_pp']), calmar_diff_median=(a['calmar_diff_median'], b['calmar_diff_median']))
out['train_summary_rep_vs_orig'] = s
out['frozen_same'] = FZ == OV['frozen_rule']['id']
# 驗證期、全期連續路徑
mp = {'CUR|fresh': ('CUR', 'fresh'), 'FZ|fresh': (FZ, 'fresh'), 'CUR|cont': ('CUR', 'cont'), 'FZ|cont': (FZ, 'cont'),
      'CUR|fresh_blk': ('CUR', 'blk'), 'FZ|fresh_blk': (FZ, 'blk')}
vp = {}
for nm, (c, mode) in mp.items():
    D = OV['paths'][nm]
    src = {'fresh': R['paths_fresh'], 'cont': R['paths_cont'], 'blk': R['paths_fresh_blk']}[mode][c]
    mine = {k: {f: src[k][f] for f in D[k] if f in src[k]} for k in D}
    vp[nm] = cmp(mine, D)
out['valid_cont_paths'] = vp
out['verdict_same'] = all(R['verdict'][k] == OV['verdict'][k] for k in ('T0', 'V1', 'V2', 'V3', 'V4', 'overall'))
out['random_registered_max_abs_diff_pp'] = max(abs(a - b) for a, b in zip(
    R['random_control']['registered_seeds_20261116_plus_i']['fresh_d_q10_50_90'] + R['random_control']['registered_seeds_20261116_plus_i']['cont_d_q10_50_90'] +
    R['random_control']['registered_seeds_20261116_plus_i']['fresh_dd_q10_50_90'] + R['random_control']['registered_seeds_20261116_plus_i']['cont_dd_q10_50_90'],
    OV['random_control']['summary']['fresh']['random_valid_diff_pp_q10_50_90'] + OV['random_control']['summary']['cont']['random_since2019_diff_pp_q10_50_90'] +
    OV['random_control']['summary']['fresh']['random_dd_diff_pp_q10_50_90'] + OV['random_control']['summary']['cont']['random_dd_diff_pp_q10_50_90']))
# 事後：其他候選驗證窗
ph = {}
for c, x in OP['candidates_valid_and_cont'].items():
    m = R['posthoc_candidates_valid'][c]; cp = R['cont_paths']['pairs'][c]
    mine = dict(valid_cagr=m['valid']['median_pp'], valid_dd=m['valid_dd']['median_pp'], fresh_seg2022=m['seg']['2022']['median_pp'],
                fresh_seg2407=m['seg']['2024-07~2025-04']['median_pp'], cont_since2019=cp['since2019']['median_pp'], cont_full_dd=cp['maxdd']['median_pp'],
                cont_seg2018=cp['seg']['2018']['median_pp'], cont_seg2020=cp['seg']['2020-03']['median_pp'], cont_seg2022=cp['seg']['2022']['median_pp'],
                cont_seg2407=cp['seg']['2024-07~2025-04']['median_pp'])
    npos = dict(valid_cagr=m['valid']['npos'], valid_dd=m['valid_dd']['npos'], fresh_seg2022=m['seg']['2022']['npos'], fresh_seg2407=m['seg']['2024-07~2025-04']['npos'])
    ph[c] = dict(max_abs_diff_pp=max(abs(mine[k] - x[k]['median_pp']) for k in mine),
                 npos_same=all(npos[k] == x[k]['npos'] for k in npos))
out['posthoc_candidates'] = ph
sf = R['main_CUR_cont_stockfrac']
out['stockfrac'] = dict(rep_mean_since2019=sf['mean_since2019_after_20190101'], orig=OP['CUR_stockfrac_mean_since2019'],
                        rep_seg=sf['seg_mean'], orig_seg=OP['CUR_stockfrac_mean_seg'])
R['comparison'] = out
pp = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s23rep/prep_rep.json')
if pp.exists():
    pr = json.loads(pp.read_text())
    R['step0_prep'] = {k: pr[k] for k in ('thr_minp1', 'thr_minp3', 'rule_effect_shift0', 'rule_effect_train_by_shift', 'meta')}
    R['step0_prep']['note'] = 'BRD3 不足 3 列時用現有列（minp=1）才與 PREREG 門檻一致；觸發月份與 data_check 逐月相同'
(H / 'results_rep.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1))
