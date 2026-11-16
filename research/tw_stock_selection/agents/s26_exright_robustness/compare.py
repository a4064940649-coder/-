#!/usr/bin/env python3
"""S26 彙整：各題「修正前」（原資料夾的 results*.json，未改）對「修正後」（runx.py 在 cslib.EXRIGHT_FIX=True 下原封重跑、輸出在 scratchpad）的
判定用主比較，逐題列出數字與判定是否改變；另把判定欄位（字串／布林）全部逐項比對（generic_verdict_changes）。
也算 risk_update.json（網站 risk.json／README 標「修正前」的數字）。
python3 -I -B compare.py → results.json、risk_update.json（本資料夾）。"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common, jdiff

AG, RUN = common.AG, common.SCR / 'run'
pp = lambda x: None if x is None else round(float(x) * 100, 2)
r2 = lambda x: None if x is None else round(float(x), 2)


def J(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def old(folder, f='results.json'):
    return J(AG / folder / f)


def new(tag, f='results.json'):
    return J(RUN / tag / f)


def vdiff(a, b):
    if a is None or b is None:
        return None
    o = jdiff.diff(a, b)
    return [dict(path=p, before=x, after=y) for p, x, y in o['str']]


def soft(tag):
    s = J(common.OUT / f'{tag}__soft.json')
    return None if s is None else dict(panel_exright_fix=s['panel_exright_fix'], soft_asserts=[x['msg'][:160] for x in s['soft_asserts']], secs=s['secs'], peak_mb=s['peak_mb'])


def accept_fail(res):
    out = []

    def w(x, p):
        if isinstance(x, dict):
            for k, v in x.items():
                w(v, p + '/' + str(k))
        elif x is False:
            out.append(p)
    if res and 'acceptance' in res:
        w(res['acceptance'], '')
    return out


def paired(a, b):
    k = [x for x in a if x in b]
    d = np.array([a[x] - b[x] for x in k]) * 100
    return dict(median_pp=round(float(np.median(d)), 2), npos=int((d > 0).sum()), n=len(k))


S = {}

# ---------------------------------------------------------------- S6
a, b = old('s6_satellite_sizing'), new('s6_run_fix1')


def s6(r):
    c = r['cells']; t = r['tradeoff']
    return dict(main_WN_70=dict(r19=pp(t['rows']['WN_70']['r19']), dd=pp(t['rows']['WN_70']['dd']), path_median=pp(c['WN_70']['paths20']['median'])),
                W10_70=dict(r19=pp(t['rows']['W10_70']['r19']), dd=pp(t['rows']['W10_70']['dd'])),
                frontier_since2019=t['frontier_since2019'], frontier_full=t['frontier_full'],
                WN_70_on_frontier=('WN_70' in t['frontier_since2019']),
                classes={k: v['class'] for k, v in c.items()},
                RW10_minus_W10_paths=paired(c['RW10_70']['paths20']['values'], c['W10_70']['paths20']['values']),
                RWN_minus_WN_paths=paired(c['RWN_70']['paths20']['values'], c['WN_70']['paths20']['values']),
                path_median={k: pp(v['paths20']['median']) for k, v in c.items()})


S['S6'] = dict(title='S6 衛星資金利用率與名次加權（每檔金額前緣）', main='8 格的效率前緣（2019 起／全期，單一路徑）、主格 WN_70 分類、名次加權 20 條配對差',
               before=s6(a), after=s6(b), soft=soft('s6_run_fix1'), accept_fail_after=accept_fail(b))

# ---------------------------------------------------------------- S7
a, b = old('s7_locked_base'), new('s7_run_fix1')


def s7(r):
    if r is None:
        return None
    c = r['cells']
    dl = lambda cid, d: c[cid]['delay'][str(d)]['since2019']
    return dict(main_WN_70L=dict(r19=pp(dl('WN_70L', 0)['cagr']), dd=pp(dl('WN_70L', 0)['maxdd'])),
                lock_effect={k: dict(ret_diff_pp_2019=r2(v['ret_diff_pp']['since2019']), dd_shallower_pp=r2(v['dd_shallower_pp'])) for k, v in r['lock_effect'].items()},
                delay_cost_pp={cid: {d: r2(c[cid]['delay_cost_pp'][d]['since2019']) for d in ('1', '2')} for cid in ('WN_70L', 'W10_70L', 'WN_70')},
                delay2_maxdd={cid: pp(dl(cid, 2)['maxdd']) for cid in ('WN_70L', 'W10_70L', 'WN_70')},
                frontier_a_since2019=r['tradeoff']['frontier_a_since2019'], frontier_b_since2019=r['tradeoff']['frontier_b_since2019'],
                classes={k: v['class'] for k, v in c.items()})


S['S7'] = dict(title='S7 底倉鎖定與晚下單', main='鎖定 vs 不鎖定（報酬差、回撤差）、晚 1／2 天代價、前緣、分類',
               before=s7(a), after=s7(b), soft=soft('s7_run_fix1'), accept_fail_after=accept_fail(b))

# ---------------------------------------------------------------- S8
a, b = old('s8_drawdown_cap'), new('s8_run_fix1')


def s8(r):
    if r is None:
        return None
    c = r['cells']; v = r['vs_WN_70L']
    return dict(main_verdict=r['main_verdict'],
                main_C12=dict(r19=pp(c['WN_70L_C12']['runs']['strict']['since2019']['cagr']) if 'runs' in c['WN_70L_C12'] and 'strict' in c['WN_70L_C12']['runs'] else None,
                              dd_shallower_pp=r2(v['WN_70L_C12']['dd_shallower_pp']), keep_2019=r2(v['WN_70L_C12']['keep_ratio']['since2019'] * 100),
                              paths_dd_shallower_n=v['WN_70L_C12']['paths_dd_shallower_n']),
                paths_dd_median={k: pp(x['paths20']['maxdd_median']) for k, x in c.items()},
                paths_n_dd_within_36={k: x['paths20'].get('n_dd_within_36') for k, x in c.items()},
                frontier_a_since2019=r['tradeoff']['frontier_a_since2019'], frontier_b_since2019=r['tradeoff']['frontier_b_since2019'],
                classes={k: x['class'] for k, x in c.items()})


S['S8'] = dict(title='S8 壓回撤（單檔上限、衛星比例）', main='主格 WN_70L_C12 三項（回撤 ≥ −36%、保住率 ≥ 75%、20 條 ≥ 15 條變淺）→「能／不能」',
               before=s8(a), after=s8(b), soft=soft('s8_run_fix1'), accept_fail_after=accept_fail(b))

# S8 複製員的回撤公式：回撤中位數 ≈ a + b × 每檔占總資產（17 格）。x 用複製員的每檔權重（金額規則決定，修正前後幾乎不變），y 用各格 20 條回撤中位數。
rep = J(AG / 's8_drawdown_cap' / 'replicate' / 'my_numbers.json')
if rep:
    tab = rep['reviewB']['table']
    srcmap = {}
    for cid in tab:
        if cid in ('W10_70', 'WN_70', 'WN_100', 'W7_70', 'W10_100', 'W7_100'):
            srcmap[cid] = ('s6', cid)
        elif cid in ('WN_70L_C12', 'WN_70L_C10', 'WN_70L_C08', 'WN_60L', 'WN_60L_C10'):
            srcmap[cid] = ('s8', cid)
        else:
            srcmap[cid] = ('s7', cid)
    R = {'s6': (old('s6_satellite_sizing'), new('s6_run_fix1')), 's7': (old('s7_locked_base'), new('s7_run_fix1')), 's8': (old('s8_drawdown_cap'), new('s8_run_fix1'))}

    def fit(which):
        xs, ys, ok = [], [], True
        for cid, (src, k) in srcmap.items():
            r = R[src][which]
            if r is None:
                ok = False; break
            xs.append(tab[cid]['per_stock_w_daily']); ys.append(r['cells'][k]['paths20']['maxdd_median'] * 100)
        if not ok:
            return None
        x, y = np.array(xs), np.array(ys)
        bb = np.polyfit(x, y, 1); yh = np.polyval(bb, x)
        rr = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return dict(intercept_pct=round(float(bb[1]), 2), slope=round(float(bb[0]), 3), r2=round(float(rr), 3), n=len(x),
                    weight_for_minus36=round(float((-36 - bb[1]) / bb[0]), 2))
    S['S8']['formula'] = dict(desc='回撤中位數（%）≈ 截距 + 斜率 × 每檔占總資產（%）；17 格；x＝複製員 my_numbers.json 的每檔權重（修正前，金額規則決定），y＝研究員引擎 20 條回撤中位數',
                              published='−23.1% − 1.58 ×（R² 0.935；每檔 ≤ 約 8.2% 才在 −36% 內）', before=fit(0), after=fit(1))

# ---------------------------------------------------------------- S9
a, b = old('s9_signal_timing'), new('s9_run_fix1')
LAG = {'A10': 0.0, 'A11': 0.6201550387596899, 'A12': 1.2945736434108528, 'A13': 1.9767441860465116, 'A15': 3.356589147286822}   # 複製員 my_numbers.json calendar.mean_lag（日曆，不受價格影響）


def ontime(r, cid):
    """複製員口徑：A10～A15 開盤 100 條路徑，x＝定義平均延遲＋位移，迴歸截距＝準時（延遲 0）可預期。"""
    xs, ys = [], []
    for d, lag in LAG.items():
        for k, v in r['combos'][f'{d}|open|{cid}']['paths'].items():
            xs.append(lag + int(k.split('+')[1])); ys.append(v * 100)
    bb = np.polyfit(xs, ys, 1)
    return dict(intercept=round(float(bb[1]), 2), slope_pp_per_day=round(float(bb[0]), 2))


def s9(r):
    v = r['verdict']; f = r['family']
    o = dict(standard_is_best=v['standard_is_best'], A10_rank=v['standard_rank_open'], selection_bias=v['selection_bias'],
             A10_minus_median_pp={k: r2(x) for k, x in v['A10_minus_median_pp'].items()}, robust=v['robust'], robust_fails=v['robust_fails'],
             exec=v['exec_recommendation'], n_class_pass=sum(1 for x in v['all_combos_class'].values() if x == '值得上XQ測'))
    for cid in ('W10_70', 'WN_70'):
        s1, s3 = f[f'{cid}|open']['S1'], f[f'{cid}|open']['S3']
        o[cid] = dict(A10_r19=pp(s1['r19_max'] if v['standard_rank_open'][cid] == 1 else None),
                      fam140_median=pp(s3['r19_median']), fam140_p10=pp(s3['r19_p10']), fam140_dd_median=pp(s3['maxdd_median']), fam140_dd_p10=pp(s3['maxdd_p10']),
                      fam100_A10_A15_median=pp(s3['fam2_r19_median']), fam100_p10=pp(s3['fam2_r19_p10']),
                      fam100_dd_median=pp(s3['fam2_maxdd_median']), fam100_dd_p10=pp(s3['fam2_maxdd_p10']),
                      A10_paths20_median=pp(s3['A10_paths_r19_median']), A10_paths20_dd_median=pp(s3['A10_paths_maxdd_median']),
                      close_minus_open_median_pp=r2(r['S4_exec'][cid]['median_pp']), close_better_n=r['S4_exec'][cid]['n_positive'],
                      ontime_regression=ontime(r, cid), n_above_0050_140=s3['n_above_0050'])
    return o


S['S9'] = dict(title='S9 換股時效與訊號日家族', main='A10 在 7 個定義排第幾、A10 − 家族中位數 ≥ 3 → 偏樂觀、開盤 vs 收盤、28 組合八項',
               before=s9(a), after=s9(b), generic_verdict_changes=vdiff(a['verdict'], b['verdict']), soft=soft('s9_run_fix1'), accept_fail_after=accept_fail(b))

# ---------------------------------------------------------------- S10
a, b = old('s10_exit_details'), new('s10_run_fix1')


def s10(r):
    v = r['verdict']; p = r['pairs']
    pick = lambda k: {x: (r2(p[k][x]) if isinstance(p[k][x], float) else p[k][x]) for x in ('verdict_ret', 'npos', 'median_pp', 'dd_verdict', 'dd_n_shallower', 'dd_median_pp') if x in p[k]}
    return dict(E0_is_best=v['E0_is_best'], E0_rank_path_median=v['E0_rank_path_median'], E0_rank_path_dd=v['E0_rank_path_dd'],
                any_stable_better=v['any_stable_better'], trailing_stop_cuts_dd=v['trailing_stop_cuts_dd'], trailing_detail=v['trailing_detail'],
                MANO_W10_70=pick('MANO|W10_70'), MANO_WN_70=pick('MANO|WN_70'), TS15_WN_70=pick('TS15|WN_70'))


S['S10'] = dict(title='S10 出場家族', main='E0 在 11 個出場的名次（20 條中位數）、哪個「穩定地比較好」、移動停損能否壓回撤',
                before=s10(a), after=s10(b), generic_verdict_changes=vdiff(a['verdict'], b['verdict']), soft=soft('s10_run_fix1'), accept_fail_after=accept_fail(b))

# ---------------------------------------------------------------- S17
a, b = old('s17_surge_definition_family'), new('s17_run_fix1')


def s17(r):
    v = r['verdict']; p = r['pairs']; f = r['family']
    return dict(STD_is_best=v['STD_is_best'], STD_rank_path_median=v['STD_rank_path_median'], STD_rank_path_dd=v['STD_rank_path_dd'],
                STD_minus_family_median_pp={k: r2(x) for k, x in v['STD_minus_family_median_pp'].items()}, optimism=v['optimism'],
                reestimate={k: (x if isinstance(x, str) else {kk: pp(vv) for kk, vv in x.items()}) for k, x in v['reestimate'].items()},
                any_stable_better=v['any_stable_better'],
                family_path_median={c: {d: pp(x) for d, x in f[c]['path_median'].items()} for c in ('W10_70', 'WN_70')},
                paths140_median={c: pp(f[c]['P1']['paths140_median']) for c in ('W10_70', 'WN_70')},
                NOMU={c: dict(npos=p[f'NOMU|{c}']['npos'], median_pp=r2(p[f'NOMU|{c}']['median_pp']), verdict=p[f'NOMU|{c}']['verdict_ret']) for c in ('W10_70', 'WN_70')})


S['S17'] = dict(title='S17 SURGE 定義家族', main='STD 在 7 個定義排第幾、STD − 家族中位數 ≥ 3 → 偏樂觀、哪個「穩定地比較好」',
                before=s17(a), after=s17(b), generic_verdict_changes=vdiff(a['verdict'], b['verdict']), soft=soft('s17_run_fix1'), accept_fail_after=accept_fail(b))

# ---------------------------------------------------------------- S19
a, b = old('s19_xq_threshold'), new('s19_run_fix1')


def s19(r):
    v = r['verdict']; out = {}
    for c in ('W10_70', 'SAT'):
        out[c] = dict(J1=v[c]['J1'], XT_minus_E0_median=r2(v[c]['XT_minus_E0']['median']), npos=v[c]['XT_minus_E0']['npos'], J2=v[c]['J2'],
                      decomposition={k: r2(x) for k, x in v[c]['decomposition'].items()}, J1_1923=v[c]['J1_1923'], J1_24=v[c]['J1_24'])
    out['J4'] = v['J4_dynamic']['verdict']
    for k in v:
        if k.startswith('J') and k not in ('J4_dynamic', 'J7_class') and isinstance(v[k], (str, bool)):
            out[k] = v[k]
    return out


S['S19'] = dict(title='S19 XQ 門檻版', main='J1 門檻版 − 名次版（20 條配對差中位數）、J2 差距來源、J4 動態門檻',
                before=s19(a), after=s19(b), generic_verdict_changes=vdiff(a['verdict'], b['verdict']), soft=soft('s19_run_fix1'), accept_fail_after=accept_fail(b))

# ---------------------------------------------------------------- 凍結規則的驗證（S20、S21、S23、S25）＋訓練期重跑看凍結選擇
for sid, folder, pre in (('S20', 's20_persistent_growth', 's20'), ('S21', 's21_industry_resonance', 's21'), ('S23', 's23_market_regime', 's23'), ('S25', 's25_factor_momentum', 's25')):
    a, b = old(folder), new(f'{pre}_validate_fix1')
    ta, tb = J(AG / folder / 'FROZEN.json'), new(f'{pre}_train_fix1_nf', 'FROZEN.json')
    rta, rtb = old(folder, 'results_train.json'), new(f'{pre}_train_fix1_nf', 'results_train.json')

    def fz(r):
        if r is None:
            return None
        v = r['verdict']; p = r['pairs']['FZ_vs_CUR|fresh']
        o = dict(frozen_rule=r['frozen_rule'] if isinstance(r['frozen_rule'], str) else r['frozen_rule'].get('id'),
                 T0=v['T0'], V1=v['V1'], V2=v['V2'], V3=v['V3'], V4=v['V4'], overall=v['overall'],
                 valid_npos=p['valid']['npos'], valid_median_pp=r2(p['valid']['median_pp']),
                 valid_dd_npos=p['valid_dd']['npos'], valid_dd_median_pp=r2(p['valid_dd']['median_pp']),
                 seg={k: r2(x['median_pp']) for k, x in p['seg'].items()})
        if 'V3_block_union' in r and isinstance(r['V3_block_union'], dict):
            o['V3_block_union'] = {k: (r2(x) if isinstance(x, float) else x) for k, x in r['V3_block_union'].items() if not isinstance(x, (dict, list))}
        return o

    def tr(t, rt):
        if t is None:
            return None
        o = dict(frozen=t['frozen_rule']['id'], T0_pass=t.get('T0_pass'))
        if rt and 'summary' in rt:
            o['train_summary'] = {k: ({kk: (r2(vv) if isinstance(vv, float) else vv) for kk, vv in x.items() if not isinstance(vv, (dict, list))} if isinstance(x, dict) else x)
                                  for k, x in rt['summary'].items()}
        return o
    S[sid] = dict(title={'S20': 'S20 持續成長篩選', 'S21': 'S21 產業共振', 'S23': 'S23 市場環境減碼', 'S25': 'S25 因子動能減碼'}[sid],
                  main='凍結規則（原 FROZEN.json，不重選）在驗證窗的 T0、V1～V4 → 通過／不通過；另以修正後重跑訓練期看凍結選擇會不會變',
                  before=fz(a), after=fz(b), train_before=tr(ta, rta), train_after=tr(tb, rtb),
                  generic_verdict_changes=vdiff(a['verdict'], b['verdict']) if b else None,
                  soft=soft(f'{pre}_validate_fix1'), soft_train=soft(f'{pre}_train_fix1_nf'), accept_fail_after=accept_fail(b))

# ---------------------------------------------------------------- S13（凍結規則驗證，候選層級）
a, b = old('s13_big_winner_features'), new('s13_validate_fix1')
ta, tb = J(AG / 's13_big_winner_features' / 'FROZEN.json'), new('s13_train_fix1_nf', 'FROZEN.json')


def s13(r):
    if r is None:
        return None
    v = r['validation']
    return dict(PASS=v['PASS'], checks={k: dict(ok=x['ok'], **{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in x.items() if kk not in ('ok', 'need', 'top5_removed')}) for k, x in v['checks'].items()},
                n=v['n'], keep_share=round(v['keep_share'], 4))


S['S13'] = dict(title='S13 大贏家特徵', main='凍結規則（ADVCHG ≥ 0.296）驗證期 V1～V4 → PASS',
                before=s13(a), after=s13(b),
                train_before=None if ta is None else dict(feature=ta['feature'], threshold=ta['threshold'], training_weak=ta['training_weak']),
                train_after=None if tb is None else dict(feature=tb['feature'], threshold=tb['threshold'], training_weak=tb['training_weak']),
                soft=soft('s13_validate_fix1'), soft_train=soft('s13_train_fix1_nf'))

# ---------------------------------------------------------------- 其他 run.py 型（S11、S12、S14、S15、S16）：判定欄位逐項比對
for sid, folder, tag, title, main in (
        ('S11', 's11_pead_second_signal', 's11_run_fix1', 'S11 季報驚喜第二訊號', '相關程度、Q 對 0050、C 是否有分散效果、XQ 建議'),
        ('S12', 's12_surge_size_tiers', 's12_run_fix1', 'S12 SURGE 大小型分層', '大型股是否有效、選股超額、Q55 是否降低依賴'),
        ('S14', 's14_corporate_events', 's14_run_fix1', 'S14 公司事件', 'D 對 0050、C 補充價值、XQ 建議'),
        ('S15', 's15_strong_pullback', 's15_run_fix1', 'S15 強勢股拉回', 'P 對 0050、C 補充價值、XQ 建議'),
        ('S16', 's16_surge_otc', 's16_run_fix1', 'S16 加上櫃', '加上櫃 vs 只上市、上櫃單獨、XQ 建議')):
    a, b = old(folder), new(tag)
    keep = lambda v: None if v is None else {k: x for k, x in v.items() if k not in ('fails', 'class_fails')}
    S[sid] = dict(title=title, main=main, before=keep(a['verdict']), after=keep(b['verdict']) if b else None,
                  generic_verdict_changes=vdiff(a['verdict'], b['verdict']) if b else None, soft=soft(tag), accept_fail_after=accept_fail(b))

# ---------------------------------------------------------------- S24、S25 describe（描述性，非判定）
a, b = old('s24_return_attribution'), new('s24_fix1')
S['S24'] = dict(title='S24 報酬歸因（描述，無判定規則）', main='CAPM β／α、主模型 α（t）',
                before=a, after=b, soft=soft('s24_fix1'))
a, b = old('s25_factor_momentum', 'results_describe.json'), new('s25_describe_fix1', 'results_describe.json')
S['S25_describe'] = dict(title='S25 扣同業超額（描述）', main='每筆扣同業籃子超額（t）', before=a, after=b, soft=soft('s25_describe_fix1'))

out = dict(meta=dict(desc='S26：除權息修正（cslib.EXRIGHT_FIX=True）後重跑 S6～S25 判定用主比較。修正前＝原資料夾 results.json（未改）；修正後＝runx.py 原封執行各題腳本。',
                     harness_check='s10 run.py 在修正關閉下重跑，與原 results.json 逐位相同（只差 runtime_s）',
                     not_run={'S18': '名單產生器驗收（make_list 對照回測），不是判定；修正後名單已由 data_exright_fix／exright_engine_fix 核對（2026-11 名單不變，F1 11／129 個月變動）',
                              'S22': '資料不足，沒有算報酬，沒有判定可重驗'}),
           studies=S)
common.dump('results_full.json', out)
print('studies', list(S))
