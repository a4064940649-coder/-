"""前瞻預測名單（G8／P1 前瞻）：--asof 截斷資料，只用 asof 當日（含）以前的收盤、PIT 股數、殖利率。
輸出 live_tracking/<etf>_<review>_pred_asof<asof>.csv 與 forward_log.json（建立時間戳必須早於截止日）。
- 臺灣50 2026-12 審核：截止 2026-11-23（生效 2026-12-18 前 25 日的週一），公告 2026-12-04，生效 2026-12-18 收盤。規則版本＝P1 凍結 v4。
- 0056 2026-12 審核：同日程；P1 未過（recall@35 只有 0.32），只做 P1 前瞻檢驗，不交易。殖利率代理＝y4。
- 00878 2026-11（MSCI ESG 分數不公開）、00919 2026-12（上櫃股與規則未重建）：記錄「無法預測」。
用法：python3 -I -B s6_forward.py --asof 2026-10-22（截止日當晚再跑一次 --asof 2026-11-23 作為正式凍結名單）。"""
import sys, json, argparse, hashlib, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import eflib as E
ap = argparse.ArgumentParser(); ap.add_argument('--asof', required=True); a = ap.parse_args()
_R = json.load(open(E.HERE / 'results.json'))
assert 'PREREG' in _R, '必須先凍結 PREREG（s7_prereg.py）才能產生前瞻名單'
asof = pd.Timestamp(a.asof)
# 截斷：所有面板只保留 asof 以前
for k in ('close_raw', 'adj', 'open', 'value', 'src'):
    E._c[k] = E.P(k).loc[:asof]
import s2_tw50_p1 as S2, s3_0056_p1 as S3
S3.pe = S3.pe[S3.pe.date <= asof]; S3.xd = S3.xd[S3.xd.date <= asof]
d = E.tday_le(asof)
assert d <= asof
SCHED = dict(cutoff='2026-11-23', ann='2026-12-04', eff='2026-12-18')
now = dt.datetime.now(dt.timezone.utc)
assert now < pd.Timestamp(SCHED['cutoff'] + ' 13:30', tz='Asia/Taipei').tz_convert('UTC'), '建立時間必須早於截止日收盤'
OUT = E.HERE / 'live_tracking'; OUT.mkdir(exist_ok=True)
log = {'created_utc': now.isoformat(), 'asof': a.asof, 'data_last_date': str(d.date()), 'schedule_2026_12': SCHED, 'files': {}}
# 臺灣50
adds, dels, rk, mem = S2.predict(d, patch=True)
rows = [dict(etf='TW50', review='2026-12', side='add', code=c, mcap_rank=int(rk[c])) for c in adds] + \
       [dict(etf='TW50', review='2026-12', side='del', code=c, mcap_rank=(int(rk[c]) if c in rk.index else None)) for c in dels]
near = [dict(etf='TW50', review='2026-12', side='watch_add_41_45', code=c, mcap_rank=int(rk[c])) for c in rk.index if c not in mem and 41 <= rk[c] <= 45] + \
       [dict(etf='TW50', review='2026-12', side='watch_del_56_60', code=c, mcap_rank=int(rk[c])) for c in mem if c in rk.index and 56 <= rk[c] <= 60]
f1 = OUT / f'tw50_2026-12_pred_asof{a.asof}.csv'; pd.DataFrame(rows + near).to_csv(f1, index=False)
# 0056（只做 P1 前瞻）
U, Y, rk2 = S3.yields(d)
y = Y['y4'].rank(ascending=False, method='first').dropna().sort_values()
f2 = OUT / f'0056_2026-12_y4rank_asof{a.asof}.csv'
pd.DataFrame({'etf': '0056', 'review': '2026-12', 'code': y.index, 'y4_rank': y.values.astype(int), 'y4': Y['y4'].reindex(y.index).round(3).values,
              'mcap_rank': [int(rk2[c]) for c in y.index]}).to_csv(f2, index=False)
for f in (f1, f2):
    log['files'][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
log['not_predictable'] = {'00878 2026-11': 'MSCI ESG 分數不公開，規則無法重建', '00919 2026-12': '母體含上櫃股（2025-09 前無上櫃日 K），規則未重建；名單完整率 1/8'}
lp = OUT / 'forward_log.json'
L = json.load(open(lp)) if lp.exists() else []
L.append(log); json.dump(L, open(lp, 'w'), ensure_ascii=False, indent=1)
print(json.dumps(log, ensure_ascii=False, indent=1)); print('TW50 add', adds, 'del', dels)
