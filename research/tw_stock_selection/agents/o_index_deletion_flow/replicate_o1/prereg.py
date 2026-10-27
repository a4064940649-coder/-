"""步驟 5：複製員的凍結紀錄（只能跑一次）。寫下要照抄的 V6 規格與判定條件，算 sha256；已存在就拒絕覆寫。"""
import sys, json, hashlib, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rlib as L
f = L.HERE / 'PREREG_replica.json'
if f.exists():
    sys.exit('PREREG_replica.json 已存在，不重跑')
spec = {
    'source_spec_sha256_researcher': '9dd08c248726fb39340c229566022be91a0340d90216765c67f0fbe7dbce35c9',
    'events_sha256': L.FROZEN_SHA,
    'classes_K': L.KCLS, 'cut': L.CUT, 'hold': L.H, 'entry': 'E close', 'exit': 'E+20 close',
    'filters': 'in panel; E-1 adv60>=1e7 (min 40 obs on >=300-name days); E-1 close>=10; E traded; E close < prev*(1.065 | 1.095 from 2015-06-01); E not in 2024-02-15..2025-09-23',
    'bench': 'equal-weight all listed stocks traded at E, buy-and-hold E close->E+20 close (ffill for halted/delisted)',
    'cost': {'normal': L.COST, 'stress': L.STRESS},
    'windows': {'select': [str(L.SEL[0].date()), str(L.SEL[1].date())], 'validate': [str(L.VAL[0].date()), str(L.VAL[1].date())],
                'holdout': [str(L.HOLD[0].date()), str(L.HOLD[1].date())]},
    'verdict': '(i) high net mean>0, t_month>=2, median>0, trim5>0; (ii) H-L paired>0, t>=1.5; (iii) 2019-2020 and 2021+ both>0; (iv) monthly corr with main & disposal <0.3; (v) 70/20/10 t >= 80/20 t and DD not deeper',
    'sleeve': '10 slots, NAV/10 per trade, cap 5% of E-1 adv20, idle in 0050 (fee 0.0855% each way, +0.1% tax on sell), same-day higher pressure first',
    'p4': 'main EXFIN n10 gapfix official plan; MSCI STD/SC or TW50 add with e_k<=E<e_{k+1}-1; paired diff E close -> e_{k+1} open vs other holdings; adopt if mean<-1%, t<=-2, both 2006-2015 & 2016-2026 <0',
}
txt = json.dumps(spec, ensure_ascii=False, sort_keys=True)
rec = {'spec': spec, 'sha256': hashlib.sha256(txt.encode()).hexdigest(), 'utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
       'note': '在算任何 2019 年後報酬之前寫入'}
f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
print(rec['sha256'], rec['utc'])
