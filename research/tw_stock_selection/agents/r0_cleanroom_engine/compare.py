# -*- coding: utf-8 -*-
"""R0 對照：清潔室結果 vs 正式結果（res_official.json、navs_official.csv.gz）。用法：python3 -I -B compare.py <前綴> <輸出json>"""
import sys, json, gzip, csv, datetime as dt
OFF = '/home/user/-/research/tw_stock_selection/agents/official_nav/out/res_official.json'
NAV = '/home/user/-/xq_scripts/site/navs_official.csv.gz'
def main(prefix, out):
    off = json.load(open(OFF))['strats']['E0|W10_70']
    mine = {x['rev_month']: x for x in json.load(open(prefix + '_lists.json'))}
    summ = json.load(open(prefix + '_summary.json'))['summary']
    months = sorted(off['sig_dates'])
    rows = []; inter = union = same_date = exact = 0; nm = no = 0
    for m in months:
        o = off['targets'][m]; mi = mine.get(m, {}); f = mi.get('f1', [])
        sd = mi.get('sig_date'); same_date += sd == off['sig_dates'][m]
        a, b = set(o), set(f); i, u = len(a & b), len(a | b)
        inter += i; union += u; nm += len(b); no += len(a); exact += (a == b and list(o) == list(f))
        rows.append(dict(rev_month=m, sig_official=off['sig_dates'][m], sig_r0=sd, n_official=len(a), n_r0=len(b), n_common=i,
                         jaccard=round(i / u, 4) if u else 1.0, same_order=list(o) == list(f),
                         only_official=sorted(a - b), only_r0=sorted(b - a)))
    extra = sorted(set(mine) - set(months)); extra_nonempty = [m for m in extra if mine[m]['f1']]
    nav_o = {}
    for r in csv.DictReader(gzip.open(NAV, 'rt')): nav_o[r['date']] = float(r['E0|W10_70'])
    nav_m = {}
    for r in csv.DictReader(open(prefix + '_navs.csv')): nav_m[r['date']] = float(r['nav'])
    common = sorted(set(nav_o) & set(nav_m))
    # 日報酬相關與追蹤差
    import math
    ro = [nav_o[common[i]] / nav_o[common[i - 1]] - 1 for i in range(1, len(common))]
    rm = [nav_m[common[i]] / nav_m[common[i - 1]] - 1 for i in range(1, len(common))]
    n = len(ro); mo = sum(ro) / n; mm = sum(rm) / n
    cov = sum((a - mo) * (b - mm) for a, b in zip(ro, rm)); vo = sum((a - mo) ** 2 for a in ro); vm = sum((b - mm) ** 2 for b in rm)
    corr = cov / math.sqrt(vo * vm)
    te = math.sqrt(sum((a - b) ** 2 for a, b in zip(ro, rm)) / n) * math.sqrt(252)
    ratio = [nav_m[x] / nav_o[x] for x in common]
    st = off['std']
    def cg(nav, start):
        prev = [x for x in common if x < start]; d0 = prev[-1] if prev else common[0]; d1 = common[-1]
        y = (dt.date.fromisoformat(d1) - dt.date.fromisoformat(d0)).days / 365.25
        return (nav[d1] / nav[d0]) ** (1 / y) - 1
    res = dict(
        layer_a=dict(n_months=len(months), same_signal_date=same_date, diff=[(r['rev_month'], r['sig_official'], r['sig_r0']) for r in rows if r['sig_official'] != r['sig_r0']],
                     r0_extra_months=extra, r0_extra_months_with_f1=extra_nonempty),
        layer_b=dict(n_official_names=no, n_r0_names=nm, n_common=inter, overlap_jaccard=round(inter / union, 4),
                     recall_of_official=round(inter / no, 4), precision_of_r0=round(inter / nm, 4),
                     months_identical_set_and_order=exact, months_identical_set=sum(1 for r in rows if not r['only_official'] and not r['only_r0']),
                     months_diff=[r for r in rows if r['only_official'] or r['only_r0']]),
        layer_c=dict(official=dict(cagr_2019=round(st['cagr_since2019'] * 100, 2), cagr_full=round(st['cagr_full'] * 100, 2), cagr_2024=round(st['cagr_since2024'] * 100, 2),
                                   maxdd=round(st['maxdd'] * 100, 2), n_closed=st['n_trades_closed'], final_nav=round(st['final_nav']),
                                   yearly={k: round(v * 100, 2) for k, v in st['yearly'].items()}, dd_peak=st['dd_peak'], dd_trough=st['dd_trough']),
                     r0={k: summ[k] for k in ('cagr_2019', 'cagr_full', 'cagr_2024', 'maxdd', 'n_closed', 'final_nav', 'yearly', 'dd_peak', 'dd_trough')},
                     same_basis_common_dates=dict(n=len(common), first=common[0], last=common[-1],
                                                  cagr_2019_official=round(cg(nav_o, '2019-01-01') * 100, 2), cagr_2019_r0=round(cg(nav_m, '2019-01-01') * 100, 2),
                                                  cagr_full_official=round(cg(nav_o, '0') * 100, 2), cagr_full_r0=round(cg(nav_m, '0') * 100, 2)),
                     daily_ret_corr=round(corr, 5), tracking_error_ann_pct=round(te * 100, 2),
                     nav_ratio_min=round(min(ratio), 4), nav_ratio_max=round(max(ratio), 4), nav_ratio_last=round(ratio[-1], 4),
                     dates_only_official=len(set(nav_o) - set(nav_m)), dates_only_r0=len(set(nav_m) - set(nav_o))),
        monthly=rows)
    lc = res['layer_c']
    lc['diff_cagr_2019_pp'] = round(lc['r0']['cagr_2019'] - lc['official']['cagr_2019'], 2)
    lc['diff_cagr_full_pp'] = round(lc['r0']['cagr_full'] - lc['official']['cagr_full'], 2)
    lc['diff_maxdd_pp'] = round(lc['r0']['maxdd'] - lc['official']['maxdd'], 2)
    lc['yearly_diff_pp'] = {y: round(lc['r0']['yearly'][y] - lc['official']['yearly'][y], 2) for y in lc['official']['yearly']}
    ov = res['layer_b']['overlap_jaccard']; dc = abs(lc['diff_cagr_2019_pp'])
    res['verdict'] = '引擎獨立驗證通過' if ov >= 0.95 and dc <= 2 else ('訊號通過、撮合／資金配置待查' if ov >= 0.95 else '訊號計算有差異')
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'monthly'}, ensure_ascii=False, indent=1))
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
