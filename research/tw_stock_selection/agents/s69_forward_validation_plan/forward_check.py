# S69 前瞻驗證計畫檢查程式：讀實盤逐日淨值（策略、0050），依 PLAN_FROZEN.json 輸出目前累積超額、t、所在分區、下次檢視日。
# 用法：python3 -I -B forward_check.py <live.json 或 csv> [--plan PLAN_FROZEN.json] [--out 結果.json]
#   live.json 格式同 xq_scripts/site/data/live.json：{"rows":[{"日期":"YYYY-MM-DD","策略":數字,"0050":數字},...]}
#   csv 需有 date（或 日期）、strategy（或 策略）、0050 三欄。
# 只讀輸入檔；不改 xq_scripts。正式判定只在 10 個檢視日（每年 10 月最後交易日收盤後）做，其餘時間的輸出只供參考。
# 注意：淨值必須是含息總報酬口徑（策略與 0050 都要含現金股利）；否則跨除息日的超額會失真。
import sys, json, csv, math, hashlib, argparse, datetime
from pathlib import Path

H = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('nav'); ap.add_argument('--plan', default=str(H / 'PLAN_FROZEN.json')); ap.add_argument('--out', default=None)
a = ap.parse_args()
plan_bytes = Path(a.plan).read_bytes(); PLAN = json.loads(plan_bytes.decode('utf-8'))
plan_sha = hashlib.sha256(plan_bytes).hexdigest()
sha_file = Path(a.plan).with_suffix('.sha256')
plan_ok = sha_file.exists() and sha_file.read_text().split()[0] == plan_sha


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def load(p):
    warn = []
    if p.endswith('.json'):
        J = json.load(open(p, encoding='utf-8'))
        rows = [(r['日期'], float(r['策略']), float(r['0050'])) for r in J['rows']]
        for mt in J.get('meta', []):
            if mt.get('type') != 'code' and '未含除息' in str(mt.get('value', '')):
                warn.append('輸入檔註明價格「未含除息調整」：跨除息日後超額會失真，正式檢視前必須改成含息總報酬口徑。')
    else:
        with open(p, encoding='utf-8-sig') as f:
            R = list(csv.DictReader(f))
        g = lambda r, *ks: next(r[k] for k in ks if k in r)
        rows = [(g(r, 'date', '日期'), float(g(r, 'strategy', '策略')), float(r['0050'])) for r in R]
    rows.sort()
    assert len({d for d, _, _ in rows}) == len(rows), '日期重複'
    return rows, warn


rows, warnings = load(a.nav)
START = PLAN['start_date']; BASE = float(PLAN['start_nav']); D1 = PLAN['delta1_pp']
assert rows and rows[0][0] >= START, '第一筆早於計畫起始日'
if rows[0][0] != START:
    warnings.append(f'第一筆 {rows[0][0]} 不是起始日 {START}')
last_d, last_s, last_b = rows[-1]
cum_s, cum_b = last_s / BASE - 1, last_b / BASE - 1
days = (datetime.date.fromisoformat(last_d) - datetime.date.fromisoformat(START)).days + 1

# 日曆月：月底＝該月最後一筆；有下一個月的資料才算「已完成的月」
by_m = {}
for d, s, b in rows:
    by_m[d[:7]] = (d, s, b)
months = sorted(by_m)
exp = []; y, m = int(START[:4]), int(START[5:7])
while f'{y:04d}-{m:02d}' <= months[-1]:
    exp.append(f'{y:04d}-{m:02d}'); m += 1
    if m == 13: y, m = y + 1, 1
assert months == exp, f'月份有缺：{sorted(set(exp) - set(months))}'
done = months[:-1]
x = []; ps, pb = BASE, BASE
for mo in done:
    _, s, b = by_m[mo]
    x.append((s / ps - 1) - (b / pb - 1)); ps, pb = s, b


def stat(xs):
    n = len(xs)
    if n < 2: return None
    mu = sum(xs) / n; sd = math.sqrt(sum((v - mu) ** 2 for v in xs) / (n - 1))
    if sd == 0: return None
    se = sd / math.sqrt(n)
    return dict(n_months=n, ann_excess_pp=mu * 1200, ann_te_pp=sd * math.sqrt(12) * 100, t=mu / se, t3=(mu - D1 / 1200) / se)


def zone(lk, st):
    k = lk['look']; mm = st['ann_excess_pp']
    if st['t'] >= lk['efficacy_t_min']: return 'EFFICACY', '判定有效（越過有效界線）', '續做'
    if st['t3'] <= lk['futility_t3_max']: return 'FUTILITY', f'正式判定無效（顯著低於 +{D1:g} 個百分點）', '停用改 0050（紙上紀錄續做）'
    if lk['pragmatic_stop_if_m_lt0'] and mm < 0: return 'PRAGMATIC_STOP', '滿 5 年以上累積超額為負（務實停用線，非統計顯著）', '建議停用改 0050（紙上紀錄續做）'
    if k == 10: return ('TERMINAL_OWNER', '10 年未證實有效，點估計 ≥ +3', '擁有者決定、降低期望') if mm >= D1 else ('TERMINAL_STOP', '10 年未證實有效，點估計 < +3', '停用改 0050')
    if mm >= D1: return 'CONTINUE', '未判定，點估計 ≥ +3', '續做'
    if mm >= 0: return 'CONTINUE_LOWER', '未判定，點估計 0～+3', '續做、降低期望'
    return 'CONTINUE_WATCH', '未判定，點估計為負（未滿 5 年）', '續做、降低期望，並看 S27 警戒線'


looks_done = []; next_look = None
for lk in PLAN['looks']:
    if lk['review_month'] in done:
        st = stat(x[:lk['n_months']]); assert st and st['n_months'] == lk['n_months']
        z = zone(lk, st)
        looks_done.append(dict(look=lk['look'], review_month=lk['review_month'], **st, efficacy_t_min=lk['efficacy_t_min'], futility_t3_max=lk['futility_t3_max'], zone=z[0], meaning=z[1], action=z[2]))
    elif next_look is None:
        next_look = lk
first_dec = next((l for l in looks_done if l['zone'] in ('EFFICACY', 'FUTILITY', 'PRAGMATIC_STOP', 'TERMINAL_OWNER', 'TERMINAL_STOP')), None)
eff_any = next((l for l in looks_done if l['zone'] == 'EFFICACY'), None)


def last_weekday(ym):
    y, m = int(ym[:4]), int(ym[5:7])
    d = datetime.date(y + (m == 12), m % 12 + 1, 1) - datetime.timedelta(days=1)
    while d.weekday() >= 5: d -= datetime.timedelta(days=1)
    return d.isoformat()


cur = stat(x)
bayes = None
if x:
    T = len(x) / 12; te = cur['ann_te_pp'] if (cur and len(x) >= 12) else PLAN['planning_te_pp']
    mobs = sum(x) / len(x) * 1200; mu0, s0 = PLAN['prior_main']['mu'], PLAN['prior_main']['sd']
    pr = 1 / s0 ** 2 + T / te ** 2; pm = (mu0 / s0 ** 2 + mobs * T / te ** 2) / pr; psd = math.sqrt(1 / pr)
    bayes = dict(note='描述用，不是判定', prior=f'N({mu0:g}, {s0:g}²)', te_used_pp=te, te_source=('實際' if (cur and len(x) >= 12) else '規劃值'), post_mean_pp=pm, post_sd_pp=psd,
                 p_gt0=norm_sf(-pm / psd), p_gt3=norm_sf((3 - pm) / psd))
if len(done) < 12:
    warnings.append(f'實盤只有 {len(rows)} 個交易日、{len(done)} 個完整月：任何超額、t 值都沒有統計意義，不得據以判斷策略有沒有用。')
if not plan_ok:
    warnings.append('PLAN_FROZEN.json 與 PLAN_FROZEN.sha256 不符或缺 sha 檔：界線可能被改過，本次輸出不可當正式判定。')
if next_look and looks_done and len(done) != looks_done[-1]['n_months']:
    warnings.append('目前不是檢視日：下列「期中」數字只供參考，不得據以判定有效（多看會吃掉型一誤差）。')
out = dict(
    plan=PLAN['plan'], plan_sha256=plan_sha, plan_sha_ok=plan_ok, input=a.nav, start_date=START, last_date=last_d, trading_days=len(rows), calendar_days=days, complete_months=len(done),
    cumulative=dict(strategy_return_pct=cum_s * 100, b0050_return_pct=cum_b * 100, excess_pp=(cum_s - cum_b) * 100, relative_pct=((1 + cum_s) / (1 + cum_b) - 1) * 100),
    interim_monthly_stats=cur, bayes_descriptive=bayes, formal_looks_done=looks_done,
    status=(dict(first_decision=first_dec, efficacy_reached=bool(eff_any)) if looks_done else '尚未到第 1 次正式檢視：未判定'),
    next_review=(dict(look=next_look['look'], review_month=next_look['review_month'], date_estimate=last_weekday(next_look['review_month']) + '（該月最後一個平日；遇休市改為該月最後交易日）',
                      n_months=next_look['n_months'], efficacy_t_min=next_look['efficacy_t_min'], futility_t3_max=next_look['futility_t3_max']) if next_look else '10 次檢視已全部完成'),
    warnings=warnings)
txt = json.dumps(out, ensure_ascii=False, indent=1)
if a.out: Path(a.out).write_text(txt + '\n', encoding='utf-8')
print(txt)
