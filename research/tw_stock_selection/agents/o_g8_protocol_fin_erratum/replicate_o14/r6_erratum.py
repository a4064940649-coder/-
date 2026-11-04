# 獨立複製 o14 金融排除勘誤：只 import 共用 main_v2/v2_core（V2 引擎），分類器、計數皆自寫；不算任何報酬
import sys, json, csv, re
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/main_v2')
import v2_core as V
sys.path.pop(0)
COLS = list(V.COLS); CI = {c: j for j, c in enumerate(COLS)}; N = len(COLS)
SNAPD = Path('/home/user/-/research/tw_stock_selection/agents/o_pit_shares_industry_2006_2013/raw/industry_snap')

def read_snap(fn):
    m = {}; names = {}
    with open(SNAPD / fn, encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) < 3 or row[0] in ('UPDATE', '文件更新') or not row[0][:1].isdigit(): continue
            m[row[0]] = row[2].strip(); names[row[0]] = row[1].strip()
    return m, names
SNAPS = [('2011-07-28', 'goristock_14393dd2_grs_stock_no.csv'), ('2011-08-17', 'goristock_d9d8662c_grs_stock_no.csv'),
         ('2011-09-26', 'goristock_235da71f_grs_old_stock_no.csv'), ('2012-03-28', 'goristock_c3c718ab_grs_stock_no.csv'),
         ('2012-07-31', 'grs_893c8772_grs_stock_no.csv'), ('2013-12-18', 'grs_568e6685_tools_twse_list.csv')]
SN = [(pd.Timestamp(e),) + read_snap(f) for e, f in SNAPS]
C0 = V.FINMASK.copy()

def snapmask(s, fallback):
    _, m, _ = s
    return np.array([(m[c] in ('17',)) if c in m else fallback[j] for j, c in enumerate(COLS)])
SMASK = [snapmask(s, C0) for s in SN]
def C1(d):
    k = [i for i, s in enumerate(SN) if s[0] <= d]
    return SMASK[k[-1]] if k else C0
def C1B(d):
    k = [i for i, s in enumerate(SN) if s[0] <= d]
    return SMASK[k[-1]] if k else SMASK[0]
# C2
KW = ['金控', '金融控股', '銀行', '商銀', '保險', '產險', '人壽', '壽險', '證券', '票券', '期貨']
END = ('銀', '金', '證', '保', '壽')
EXCL = {'2049', '2947', '3049', '4576', '7819', '9917', '9925'}
allnames = {}
for c, nm in V.SL['name'].items():
    if isinstance(nm, str): allnames.setdefault(c, set()).add(nm)
for s in SN:
    for c, nm in s[2].items(): allnames.setdefault(c, set()).add(nm)
def c2fin(c):
    if c[:2] in ('28', '58', '60'): return True
    if c in EXCL: return False
    return any(any(k in nm for k in KW) or nm.endswith(END) for nm in allnames.get(c, ()))
C2M = np.array([c2fin(c) for c in COLS])

class VX(V.V2):
    def __init__(self, maskfn, *a, **k):
        self.maskfn = maskfn; super().__init__(*a, **k)
    def y3(self, d, back=0):
        y = V.K.yoy3_for_month(self.Y5 if d >= V.SWITCH else self.YL, pd.Period(d, 'M') - back)
        return np.where(self.maskfn(d), np.nan, y)

YL, Y5, _ = V.load_revenue()
A, B = pd.Timestamp('2006-01-01'), pd.Timestamp('2013-12-31')
def run(maskfn):
    v = VX(maskfn, 'gapfix', YL, Y5, exfin=False, unavail=None)
    v.sig_days = [d for d in v.sig_days if A <= d <= B]
    pl = v.plan(10, tie='official')
    return v, {d: [COLS[j] for j in p] for e, p, el, d in pl}
runs = {}
for name, fn in [('C0', lambda d: C0), ('C1', C1), ('C1B', C1B), ('C2', lambda d: C2M)]:
    runs[name] = run(fn)
v0, P0 = runs['C0']; days = sorted(P0)
out = {'n_signal': len(days), 'first': str(days[0].date()), 'last': str(days[-1].date()), 'slots': 10 * len(days)}
for k in ('C1', 'C1B', 'C2'):
    Pk = runs[k][1]; ch = []
    for d in days:
        for c in P0[d]:
            if c not in Pk[d]: ch.append((str(d.date()), c))
    out[f'changed_{k}'] = len(ch); out[f'changed_{k}_pct'] = round(100 * len(ch) / (10 * len(days)), 3); out[f'changed_{k}_list'] = ch
out['max_C1_C2_pct'] = max(out['changed_C1_pct'], out['changed_C2_pct'])
# 5854 在 C0 股池、名次
def rank_info(v, code):
    j = CI[code]; rows = []
    for d in days:
        cs = v.cross_section(d)
        if not cs['el'][j]: continue
        ordr = list(v.order(cs, 'official')[:int(cs['el'].sum())])
        rows.append((str(d.date()), ordr.index(j) + 1, int(cs['el'].sum())))
    return rows
r5854 = rank_info(v0, '5854'); rk = np.array([r[1] for r in r5854])
out['s5854'] = dict(months=len(r5854), first=r5854[0][0] if r5854 else None, last=r5854[-1][0] if r5854 else None,
                    rank_median=float(np.median(rk)), rank_p10=float(np.percentile(rk, 10)), best=min(r5854, key=lambda r: r[1]),
                    k_top10=sum('5854' in P0[d] for d in days))
# 5854 長期營收最後月份
yl = YL['5854'].dropna() if '5854' in YL.columns else None
out['s5854']['YL_last'] = str(yl.index.max()) if yl is not None and len(yl) else None
# 2841 放回
j41 = CI['2841']; m41 = C0.copy(); m41[j41] = False
v41, P41 = run(lambda d: m41)
top41 = [(str(d.date()), P41[d].index('2841') + 1) for d in days if '2841' in P41[d]]
# 以名次（完整排序）報告
def pos(v, d, code):
    cs = v.cross_section(d); o = list(v.order(cs, 'official')[:int(cs['el'].sum())]); return o.index(CI[code]) + 1 if CI[code] in o else None
out['s2841'] = dict(back_in_top10=[(dd, pos(v41, pd.Timestamp(dd), '2841')) for dd, _ in top41],
                    C1_in=[str(d.date()) for d in days if '2841' in runs['C1'][1][d]],
                    C1B_in=[str(d.date()) for d in days if '2841' in runs['C1B'][1][d]])
# 6012 與非 PIT 金融股（C0 名單中、C2 判金融但 C1 沒有 PIT 依據者）
out['c0_picks_c2fin'] = [(str(d.date()), c, P0[d].index(c) + 1) for d in days for c in P0[d] if C2M[CI[c]]]
# 無法判定股月
def undet(d, code, loose):
    if loose:
        prev = [s for s in SN if d - pd.Timedelta(days=731) <= s[0] <= d]
        nxt = [s for s in SN if d < s[0] <= d + pd.Timedelta(days=731)]
        s = prev[-1] if prev else (nxt[0] if nxt else None)
    else:
        prev = [s for s in SN if s[0] <= d]; s = prev[-1] if prev else None
    return s is None or code not in s[1]
und = {}
for loose in (True, False):
    for scope in ('top300', 'el'):
        cnt = {}
        for d in days:
            cs = v0.cross_section(d); t = cs['t']
            v60 = v0.V60[t]; px = V.K.Cn[t]
            base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
            o = np.argsort(-np.where(base, v60, -np.inf), kind='stable'); top = np.zeros(N, bool); top[o[:min(300, base.sum())]] = True; top &= base
            pool = top if scope == 'top300' else (top & np.isfinite(V.K.yoy3_for_month(YL, pd.Period(d, 'M'))))
            for j in np.where(pool & C0)[0]:
                if undet(d, COLS[j], loose): cnt[COLS[j]] = cnt.get(COLS[j], 0) + 1
        und[f"{'loose' if loose else 'strict'}_{scope}"] = dict(total=sum(cnt.values()), n_stocks=len(cnt), top15=sorted(cnt.items(), key=lambda x: -x[1])[:15])
out['undeterminable'] = und
json.dump(out, open('erratum_replica.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if not k.endswith('_list')}, ensure_ascii=False, indent=1, default=str)[:6000])
for k in ('C1', 'C1B', 'C2'): print(k, out[f'changed_{k}_list'])
print('5854 ranks', r5854)
