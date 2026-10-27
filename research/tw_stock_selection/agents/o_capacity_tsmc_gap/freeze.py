#!/usr/bin/env python3
"""凍結：讀 iterations.json（只含挑選期），依 PLAN 的事前規則機械選版本，把 PREREG（含 sha256、時間戳）寫進 results.json。
之後 oos.py 才能計算驗證期與保留期。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
f = HERE / 'results.json'
R = json.loads(f.read_text(encoding='utf-8'))
assert R.get('PLAN_sha256'), '沒有 PLAN'
if 'PREREG' in R: raise SystemExit('PREREG 已存在：只能凍結一次')
it_bytes = (HERE / 'iterations.json').read_bytes()
IT = json.loads(it_bytes)
pool = IT['pool_rule']['chosen']; w = IT['tsmc_rule']['chosen_w']

PREREG = {
 '寫於打開驗證期之前': True,
 'PLAN_sha256': R['PLAN_sha256'],
 'iterations_json_sha256': hashlib.sha256(it_bytes).hexdigest(),
 'n_versions_tried_in_SEL': IT['n_versions'],
 '選定股池': pool,
 '選定股池理由': IT['pool_rule'],
 '選定台積電權重 w': w,
 '選定權重理由': IT['tsmc_rule'],
 '凍結規格（最終版本）': {
  '名稱': f'主策略 v2 排除金融 n10（股池 {pool}）＋ 台積電固定權重 {int(w*100)}%',
  '股池': '上市股；訊號日 60 日均成交值（gapfix 窗）≥ NT$1,000 萬、收盤 ≥ 10；依 60 日均成交值排名取前 300 名（金融股參與排名）；排除金融保險業（stock_list 金融保險業＋28xx、5876、5880，以營收空值排除）；訊號日不在處置期間；隔日有開盤成交；有 52 週比與營收 3 月年增率。',
  '分數': '0.5 × pct(還原收盤 ÷ 252 日最高還原收盤，gapfix 窗) ＋ 0.5 × pct(近 3 個月營收年增率平均)，pct 在合格池內。',
  '排序與同分': '分數四捨五入到 1e-10 由高到低；同分營收 3 月年增率高者先，再代碼小者。',
  '換股': '每月第一個日期 ≥ 11 日的交易日收盤算訊號，隔日開盤成交；開盤 ≥ 跳動點漲停價且尚未持有者跳過、由下一名遞補。',
  '權重': f'換股日開盤把組合調成：2330 台積電 {int(w*100)}%；其餘 {100-int(w*100)}% 等分給主策略 10 檔（2330 若入選，權重相加）。換股之間不再平衡。',
  '成本': '一般：買 0.0855%＋滑價 0.1%；賣 0.0855%＋證交稅 0.3%＋滑價 0.1%；壓力：買賣各 0.5%。停牌持股凍結（ia_core 規則）。',
  '營收': '訊號日 < 2021-03-01 用 tw_revenue.csv；之後 rev_5y yoy（v2 BASE）。',
  '評價': '到 2026-10-12（v2 正式終點）。',
 },
 '樣本外期間': {'VAL': '訊號窗起點 2019-04-11～2026-09-11（結束於 2026-10-12）；日頻 2019-04-11 收盤後～2026-10-12', 'VAL_A': '2019-04-11～2023-12', 'HOLD（保留期）': '2024-01～2026-10-12'},
 '樣本外要算的東西（一次跑完、全部報告）': [
  '名次歸因：n10、n30 入選股四組在 SEL、VAL、2006～2013、2014～2020、2021～2026 的超額（對同池等權；另列對同組等權）、勝率、中位數、右尾；101～300 減 1～100 月序列（主檢定 n10、對同池等權）與群聚回歸',
  '股池四種 × w0：VAL、VAL_A、HOLD、S1～S4、ALL 的年化、回撤、t（對 0050、對同池等權）、對原版配對差；n30 佐證',
  '台積電 w 0／10／20／30%（股池 U300）：同上，加追蹤誤差、資訊比率、逐年；一般與壓力成本',
  '2330 歸因：分段回歸（2006～2013、2014～2020、2021～2026、SEL、VAL、HOLD、2006～2019-04、2019-04～2026）；HOLD 每月落差拆解；用 VAL 期 β 扣除後 HOLD 殘差月超額',
  '容量：三種資金規模的部位 ÷ 20 日均成交值分布（VAL、HOLD）；各名次組 ADV；大型基金換算',
  '穩健性：逐年、分產業、去掉最大貢獻股、壓力成本',
 ],
 '最終判定（照 PLAN，事前固定）': R['PLAN']['最終判定（事前登記）'],
}
s = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
R['PREREG'] = PREREG
R['PREREG_sha256'] = hashlib.sha256(s.encode()).hexdigest()
R['PREREG_written_at'] = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
f.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(R['PREREG_sha256'], R['PREREG_written_at'], pool, w)
