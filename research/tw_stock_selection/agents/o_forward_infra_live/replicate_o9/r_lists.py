"""A1/A2：28 份名單、時程、J0 缺值統計、官方日與未截斷版比較。"""
import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live/replicate_o9')
import numpy as np, pandas as pd
import rlib as L

out = {}
lists = {}
for (y, m) in [(2026, 9), (2026, 10)]:
    js = L.j_days(y, m); od = L.official_day(y, m)
    key = f'{y}{m:02d}'
    rec = dict(J=[str(d.date()) for d in js], official=str(od.date()))
    o, df = L.official_list(od)
    lists[('OFF', key)] = list(o.index)
    # 未截斷（全資料）版：直接用 MP 讀原始資料
    D = L.RAW
    data = L.MP.load_prices(D, od, L.PSUB)
    YOY, YOY3, _ = L.MP.load_revenue(D); dsp = L.MP.load_disposal(D)
    o2, *_ = L.MP.score_on(data, YOY3, dsp, od, ex_fin=True)
    rec['off_top10'] = list(o.index[:10]); rec['off_top10_untrunc'] = list(o2.index[:10])
    rec['off_same_order'] = list(o.index[:10]) == list(o2.index[:10])
    rec['off_n'] = len(o)
    for j, sd in enumerate(js):
        oj, dfj, nmiss, miss = L.j_list(sd, j)
        lists[(f'J{j}', key)] = list(oj.index)
        if j == 0:
            oa, *_ = L.j_list(sd, j, missing_as_unannounced=False)
            rec['J0_pool'] = len(oj); rec['J0_missing_in_pool'] = nmiss
            rec['J0_top10'] = list(oj.index[:10]); rec['J0_top10_if_missing_announced'] = list(oa.index[:10])
            rec['J0_out'] = sorted(set(oa.index[:10]) - set(oj.index[:10])); rec['J0_in'] = sorted(set(oj.index[:10]) - set(oa.index[:10]))
    # 成交日、凍結時間
    ex = []
    for name, sd in [('OFF', od)] + [(f'J{j}', d) for j, d in enumerate(js)]:
        i = L.CAL.get_loc(sd); nd = L.CAL[i + 1]
        frozen = pd.Timestamp(nd.date()) + pd.Timedelta(hours=6, minutes=49)
        exec_open = pd.Timestamp(nd.date()) + pd.Timedelta(hours=9)
        ex.append(dict(line=name, sd=str(sd.date()), exec=str(nd.date()), frozen=str(frozen), ts_ok=bool(frozen < exec_open)))
    rec['exec'] = ex
    out[key] = rec
# 11、12 月時程
for (y, m) in [(2026, 11), (2026, 12)]:
    js = L.j_days(y, m)
    out[f'{y}{m:02d}_sched'] = dict(J0=str(js[0].date()), official=str(L.official_day(y, m).date()), J12=str(js[12].date()),
                                     off_exec=str(L.CAL[L.CAL.get_loc(L.official_day(y, m)) + 1].date()))
pd.to_pickle(lists, L.WORK / 'lists.pkl')
(L.HERE / 'out_lists.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'exec'} for k, v in out.items()}, ensure_ascii=False, indent=1))
print(sum(e['ts_ok'] for k in ['202609', '202610'] for e in out[k]['exec']), 'ts_ok')
