#!/usr/bin/env python3
"""迭代建模（只用挑選期：錨定月 ≤ 2018-12；模擬只到 2019-01 官方訊號日收盤）。
用法：python3 -I -B iterate.py <round>；每輪的版本在 ROUNDS 內事前寫好，結果附加到 iterations.json。"""
import sys, json, time, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import st_core as C
T0 = time.time()
res = json.loads((C.HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PLAN_sha256' in res
st = C.setup()
SEL_END = pd.Period('2018-12', 'M')
SPECS = {
 'V01': dict(js=[1, 11], a='a0', b='b0', c='c0'), 'V02': dict(js=[1, 8, 15], a='a0', b='b0', c='c0'), 'V03': dict(js=[1, 6, 11, 16], a='a0', b='b0', c='c0'),
 'V04': dict(js=[1, 11], a='a1', b='b0', c='c0'), 'V05': dict(js=[1, 8, 15], a='a1', b='b0', c='c0'), 'V06': dict(js=[1, 6, 11, 16], a='a1', b='b0', c='c0'),
 'V07': dict(js=[1, 11], a='a0', b='b0', c='c1'), 'V08': dict(js=[1, 6, 11, 16], a='a0', b='b0', c='c1'),
 'V09': dict(js=[1, 11], a='a0', b='b1', c='c0'), 'V10': dict(js=[1, 6, 11, 16], a='a0', b='b1', c='c0'),
}
ROUNDS = {1: ['V01', 'V02', 'V03'], 2: ['V04', 'V05', 'V06'], 3: ['V07', 'V08'], 4: ['V09', 'V10'], 5: ['V11', 'V12']}
# 第 5 輪：看完第 1～4 輪挑選期結果後寫入（2026-10-27）
SPECS['V11'] = dict(js=[1, 6, 11, 16], a='a0', b='b1', c='c1')
SPECS['V12'] = dict(js=[1, 3, 5], a='a0', b='b0', c='c0')

off_days, off_end = C.official_list()
SD = [(m, d) for m, d in off_days if m <= SEL_END]
SEL_EVAL_END = [d for m, d in off_days if m > SEL_END][0]
W = C.windows_from(SD, SEL_EVAL_END)
COSTS = {'gross': C.GROSS, 'normal': C.NORMAL, 'stress_flat': C.FLAT, 'stress_plus03': C.STRESS}

def run_spec(spec, cost, max_m=SEL_END, t_end=None, log=False):
    pls = C.plan_tranches(spec['js'], spec['a'], spec['b'], max_m=max_m)
    t0 = min(pl[0][0] for pl in pls) - 1
    t1 = C.IDX.get_loc(t_end if t_end is not None else SEL_EVAL_END)
    reeq = None
    if spec['c'] == 'c1':
        reeq = [C.next_good(C.IDX.get_loc(d)) for m, d in off_days if (max_m is None or m <= max_m)]
    nav, navs, lg = C.tsim(pls, *cost, t0, t1, xrules=st['XR'], reeq_days=reeq, log=log)
    return nav, navs, lg, pls, t0

def distinct_names(pls):
    ts = [C.next_good(C.IDX.get_loc(d)) for m, d in SD[3:]]
    out = []
    for t in ts:
        u = set()
        for pl in pls:
            last = [q for q in pl if q[0] <= t]
            if last: u |= set(last[-1][1].tolist())
        out.append(len(u))
    return float(np.mean(out))

_BASE = {}
def baseline(cost):
    """J+1 單日版（同一引擎、同一終點）。"""
    if cost not in _BASE:
        _BASE[cost] = run_spec(dict(js=[1], a='a0', b='b0', c='c0'), COSTS[cost], log=True)
    return _BASE[cost]

def evaluate(name, spec):
    o = {'spec': spec}
    for cn in COSTS:
        nav, navs, lg, pls, t0 = run_spec(spec, COSTS[cn], log=(cn == 'normal'))
        bnav = baseline(cn)[0]
        b = C.block(nav, W, ['SEL', 'S1', 'S2a', 'Y2008'])
        bb = C.block(bnav, W, ['SEL'])['SEL']
        pr = C.paired(nav, bnav, W, 'SEL')
        x = dict(CAGR=b['SEL']['CAGR'], maxDD=b['SEL']['maxDD'], t_0050=b['SEL']['t_0050'], ex_mean_pp=b['SEL']['ex_mean_pp'], ex_sd_pp=b['SEL']['ex_sd_pp'],
                 TE_ann_pp=b['SEL']['TE_ann_pp'], S1_t=b['S1']['t_0050'], S2a_t=b['S2a']['t_0050'], Y2008_ex=b['Y2008']['ex_mean_pp'],
                 paired_vs_J1=pr, sd_change_vs_J1=b['SEL']['ex_sd_pp'] / bb['ex_sd_pp'] - 1, TE_change_vs_J1=b['SEL']['TE_ann_pp'] / bb['TE_ann_pp'] - 1,
                 J1=dict(CAGR=bb['CAGR'], t_0050=bb['t_0050'], ex_mean_pp=bb['ex_mean_pp'], ex_sd_pp=bb['ex_sd_pp'], TE_ann_pp=bb['TE_ann_pp']))
        if cn == 'normal':
            x['turnover'] = C.turnover_month(lg, nav, t0)
            x['reeq_cost_mean'] = float(np.mean([c[1] for c in lg['reeq_cost']])) if lg['reeq_cost'] else 0.0
            x['capacity'] = C.capacity(lg['trades'], nav, t0)
            bl = baseline('normal')
            x['capacity_J1'] = C.capacity(bl[2]['trades'], bl[0], bl[4])
            x['distinct_names_mean'] = distinct_names(pls)
            x['n_frozen'] = len(lg['frozen']); x['n_exits'] = len(lg['exits'])
        o[cn] = x
    return o

if __name__ == '__main__':
    assert 'PREREG' not in res, 'PREREG 已存在：不能再迭代'
    rnd = int(sys.argv[1])
    path = C.HERE / 'iterations.json'
    it = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'versions': {}, 'rounds': {}, 'note': '只用挑選期（錨定月 2006-04～2018-12），共同窗＝官方日曆窗；配對基準＝J+1。'}
    assert str(rnd) not in it['rounds'], '這一輪已經跑過'
    if rnd == 5:
        it['round5_reason'] = ('第 1～4 輪（挑選期）：a1（跳過他批持股）稀釋成 20～40 檔，均值大降（配對 −0.25～−0.54pp）；a0 的分批均值約等於各批單日版的平均，'
                               '月超額標準差只降 2～7%；c1（每月調回等額）與 b1（營收腿凍結在 J+1 橫斷面）都略好於 c0、b0。'
                               '第 5 輪兩版：V11＝T4 a0 b1 c1（把兩個略有幫助的選項疊在一起；屬於看過結果後的組合，有贏家詛咒風險，報告時標註）；'
                               'V12＝期限後第一週集中（J+1、+3、+5，a0 b0 c0）：在分批架構內直接檢驗「營收新鮮度」機制（P5 的反面）；'
                               '這個位置是依機制事前決定，不是依 J 掃描的高低挑的（掃描中 J+2～J+5 反而偏弱）。')
    rows = {}
    for nm in ROUNDS[rnd]:
        o = evaluate(nm, SPECS[nm])
        it['versions'][nm] = o; rows[nm] = o
        n = o['normal']
        print(nm, SPECS[nm], 'CAGR', round(n['CAGR'], 2), 't', round(n['t_0050'], 2), 'ex', round(n['ex_mean_pp'], 3), 'sd', round(n['ex_sd_pp'], 2),
              'pair', round(n['paired_vs_J1']['mean_pp'], 3), round(n['paired_vs_J1']['t'], 2), 'sdchg', round(n['sd_change_vs_J1'], 3),
              'tov', round(n['turnover'], 3), 'cap', round(n['capacity']['share_gt'], 3), 'J1cap', round(n['capacity_J1']['share_gt'], 3),
              'stressP', round(o['stress_flat']['paired_vs_J1']['mean_pp'], 3), 'gross pair', round(o['gross']['paired_vs_J1']['mean_pp'], 3),
              'names', round(n['distinct_names_mean'], 1), round(time.time() - T0), 's', flush=True)
    it['rounds'][str(rnd)] = dict(versions=ROUNDS[rnd], time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
    it['n_versions_total'] = len(it['versions'])
    path.write_text(json.dumps(it, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    print('done', round(time.time() - T0), 's')
