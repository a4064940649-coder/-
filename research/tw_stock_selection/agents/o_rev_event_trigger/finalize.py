#!/usr/bin/env python3
"""事後（POSTHOC，不參與任何判定；在 oos.py 之後執行一次，sha 與時間戳寫進 results.json 的 POSTHOC）：
(1) 報告口徑換算：2024～2026 年化改用 SOP 口徑（2023-12-29 收盤 → 評價終點），與 SOP「56.7% 對 59.2%」可比。
(2) OFF_N：官方混合日，但 J0 月（期限＝訊號日）的營收改用期限日當晚值（night 模式）——檢查官方 2.51 與 J+1 2.26 的落差有多少來自 AB 抽樣懲罰。
(3) main_picks_today.py 的日期文字修改建議（patch，不改原檔）。"""
import sys, json, hashlib, datetime, subprocess, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
RES_F = HERE / 'results.json'
res = json.loads(RES_F.read_text(encoding='utf-8'))
assert 'RESULTS' in res and 'POSTHOC' not in res
import fx_core as F
K, V = F.K, F.V
st = F.setup()
Q = json.loads((F.DATA / 'q_night.json').read_text(encoding='utf-8'))['q']
NV = pd.read_parquet(HERE / 'navs_oos.parquet')
out = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'finalize_sha256': F.sha(HERE / 'finalize.py'),
       'note': '事後計算，不參與判定'}
def cagr_from(nav, a):
    s = nav.dropna(); s = s[s.index >= pd.Timestamp(a)]
    base = nav.dropna()[nav.dropna().index <= pd.Timestamp(a)].iloc[-1]
    yrs = (s.index[-1] - pd.Timestamp(a)).days / 365.25
    return float((s.iloc[-1] / base) ** (1 / yrs) - 1) * 100, str(s.index[-1].date())
H = {}
for k in ('J+1|AB', 'J0|AB', 'J0N|AB', 'OFF|AB', 'J+1|AB_stress', 'OFF|AB_stress'):
    c, e = cagr_from(NV[k], '2023-12-29'); e50, _ = cagr_from(F.E50.reindex(NV[k].dropna().index), '2023-12-29')
    H[k] = {'CAGR_from_2023-12-29': c, 'end': e, '0050_same_span': e50}
out['HOLD_SOP_convention'] = H
# (2) OFF_N
off, end = F.official_days()
def U_offn(seed):
    Ud = V.make_unavail('mc', seed); Un = F.make_unavail('night', seed, q=Q)
    def U(d):
        c = V.deadline_case(d)
        return Un(d) if c == 'D=S' else Ud(d)
    return U
rets = []
for s in range(40):
    S, pl, a, b, sd = F.build(off, end, U_offn(s))
    n_, _ = F.sim(pl, a, b, 'normal'); rets.append(n_.pct_change())
offn = (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
W = F.windows([d for _, d in off], end)
out['OFF_N'] = {'block': F.block(offn, W, ('ALL', 'SEL', 'VAL', 'HOLD')), 'paired_vs_OFF_AB_V4': {k: F.paired_cal(offn, NV['OFF|AB'].dropna(), k) for k in ('ALL', 'SEL', 'POST')},
                'paired_vs_J+1': {k: F.paired_cal(offn, NV['J+1|AB'].dropna(), k) for k in ('ALL', 'SEL', 'POST')}}
# (3) patch
src = Path('/home/user/-/research/tw_stock_selection/main_picks_today.py')
tmp = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/ore_patch'); tmp.mkdir(parents=True, exist_ok=True)
a_ = tmp / 'a_main_picks_today.py'; b_ = tmp / 'b_main_picks_today.py'
shutil.copy(src, a_); s = src.read_text(encoding='utf-8')
rep = [
 ("  - 換股日＝每月第一個「日期 ≥ 11 日」的交易日。當天收盤後算訊號，隔一個交易日開盤買。",
  "  - 訊號日（J+1）＝每月營收法定期限日 D（10 日；遇假日順延到下一個交易日；主管機關延長時用延長後期限）之後的第一個交易日。\n"
  "    訊號日收盤後算訊號；成交日＝訊號日的下一個交易日開盤。（10 日是交易日時與舊規則「第一個 ≥ 11 日的交易日」相同；\n"
  "    10 日遇假日時比舊規則晚一個交易日，避免用到期限日當晚才公告的營收。）"),
 ("def rebalance_day(cal, y, m):\n    d = cal[(cal.year == y) & (cal.month == m) & (cal.day >= 11)]\n    return d[0] if len(d) else None",
  "SPECIAL_DEADLINE = {(2019, 2): 15, (2023, 4): 12}   # 主管機關延長營收申報期限的月份（日）；新的延長公告要手動加入\n\n"
  "def rebalance_day(cal, y, m):\n    \"\"\"訊號日 J+1：期限日 D（m 月 10 日起第一個交易日）之後的第一個交易日。\"\"\"\n"
  "    dl = SPECIAL_DEADLINE.get((y, m), 10)\n    d = cal[(cal.year == y) & (cal.month == m) & (cal.day >= dl)]\n"
  "    if len(d) == 0: return None\n    nxt = cal[cal > d[0]]\n    return nxt[0] if len(nxt) else None"),
 ("    d = pd.Timestamp(year=y, month=m, day=11)\n    while d.weekday() >= 5: d += pd.Timedelta(days=1)\n    return d",
  "    d = pd.Timestamp(year=y, month=m, day=SPECIAL_DEADLINE.get((y, m), 10))\n    while d.weekday() >= 5: d += pd.Timedelta(days=1)\n"
  "    d += pd.Timedelta(days=1)\n    while d.weekday() >= 5: d += pd.Timedelta(days=1)\n    return d"),
 ("\"\"\"t 之後的下一個換股日；資料日曆還沒涵蓋時用「第一個 ≥ 11 日的平日」預估（第二個值為 True）。\"\"\"",
  "\"\"\"t 之後的下一個訊號日；資料日曆還沒涵蓋時用「10 日起第一個平日之後的下一個平日」預估（第二個值為 True）。\"\"\""),
 ("('（預估：該月第一個 ≥ 11 日的平日，未扣除國定假日）' if est else '')",
  "('（預估：期限日＝10 日起第一個平日，訊號日＝其後下一個平日；未扣除國定假日）' if est else '')"),
]
for o_, n_ in rep:
    assert s.count(o_) == 1, o_[:40]
    s = s.replace(o_, n_)
b_.write_text(s, encoding='utf-8')
r = subprocess.run(['diff', '-u', '--label', 'a/research/tw_stock_selection/main_picks_today.py', '--label', 'b/research/tw_stock_selection/main_picks_today.py', str(a_), str(b_)], capture_output=True, text=True)
(HERE / 'main_picks_today_J1.patch').write_text(r.stdout, encoding='utf-8')
out['patch_sha256'] = F.sha(HERE / 'main_picks_today_J1.patch')
# 自我檢查：patch 版的訊號日與研究定義在 2006～2026 每個錨定月相同（用加權指數日曆）
ns = {'pd': pd}
exec(compile(s.split('# ---------------- 選股')[0].split('# ---------------- 日曆 ----------------')[1], 'x', 'exec'), ns)
cal = pd.DatetimeIndex(F.TCAL)
mism = []
for m in st['anchors']:
    want = F.TCAL[F.TCAL.get_loc(F.deadline(m)) + 1]
    got = ns['rebalance_day'](cal, m.year, m.month)
    if got != want: mism.append((str(m), str(got), str(want)))
out['patch_calendar_check'] = {'n_anchor_months': len(st['anchors']), 'mismatch': mism}
res['POSTHOC'] = F.jsonable(out)
RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(F.jsonable({k: v for k, v in out.items()}), ensure_ascii=False, indent=1)[:6000])
