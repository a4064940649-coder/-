#!/usr/bin/env python3
"""事前計畫（在 iterate.py 跑任何報酬之前寫入 plan.json，含 sha256 與時間戳）。
閘門已不通過（gate.json）→ 後備題：J0／J+1 固定換股日正式化與前視清理。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
gate = json.loads((HERE / 'gate.json').read_text(encoding='utf-8'))
assert not gate['gate_coverage_pass']
PLAN = {
 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
 'gate_json_sha256': sha(HERE / 'gate.json'), 'explore_json_sha256': sha(HERE / 'explore.json'),
 'gate_result': {'coverage_strict': gate['coverage']['strict'], 'coverage_loose': gate['coverage']['loose'],
                 'upper_bound_after_source_gap': gate['coverage']['upper_bound_after_source_gap'], 'threshold': 0.80},
 'topic': '後備題：J0／J+1 固定換股日正式化與前視清理（V4 凍結資料）',
 'candidates': {
   'J+1': '訊號日＝法定期限日 D（第 m 月 10 日起第一個交易日；2019-02→02-15、2023-04→04-12）的次一交易日；當天收盤算訊號；成交日＝次一面板交易日開盤',
   'J0': '訊號日＝D；D 收盤算訊號（只用 D 收盤前已公告營收）；成交日＝D 的次一面板交易日開盤',
   'J0N': '「期限日盤後公告完成率 ≥ 99% 的最早日」：2026-09 實測 D 當晚（次日 09:00 前）非金融完成率 99.67% → 訊號日＝D，價格腿用 D 收盤、營收腿用 D 當晚已公告值；成交日＝D 的次一面板交易日開盤'},
 'availability_model': 'fx_core.make_unavail：有時間戳（鉅亨網新聞早於訊號日）就用時間戳（ABN 版），否則 rev_announce 產業延後機率 p（金融 p＝1）；J0N 用 q＝D 當晚未公告率（data/q_night.json）；2026 INS15（含 2207、2905）期限 15 日；2020-02 疫情延長月 D < s ≤ 2020-02-14 抽 p。AB＝40 seeds（0～39），日報酬平均成一條淨值；另報逐 seed。',
 'iterations_planned': [
   'it1 J+1：非 AB 與 AB', 'it2 J0：非 AB 與 AB', 'it3 J0N：非 AB 與 AB（q 模型）', 'it4 J0 ABN（新聞時間戳＋條件機率）',
   'it5 官方混合日（OFF）AB：參考，不是候選', 'it6 J0 AB worst（期限日全部未公告）：敏感度'],
 'selection_period': '錨定月 2006-04～2018-12（訊號窗起點）；淨值模擬只到 2019-01 錨定月的同型訊號日收盤（不計算 2019 年後任何數字）',
 'metrics': {
   'lookahead_pp': '|日曆月配對差（非 AB − AB）均值|，一般成本，挑選期，pp／月',
   't_select': 'AB、一般成本、對 0050 含息的訊號窗月超額 t（挑選期）',
   'paired_vs_J1': 'AB、一般成本、日曆月配對差（候選 − J+1）的 t（挑選期）；錨定月配對差當次要指標'},
 'decision_rule': [
   '1) 前視最少的候選＝lookahead < 0.03pp／月者；若都不到，取 lookahead 最小者',
   '2) 其中取 t_select 最高者為暫定',
   '3) 暫定不是 J+1 時：若 |paired_vs_J1 t| < 2 → 選 J+1（較保守、沒有前視）',
   '4) 最終判定：只改定義、不改預期 → 可用；若選中版本（AB）全期對 0050 訊號窗 t 比官方 AB_V4 的 2.512 低 ≥ 0.1 → 「預期要下修」',
   '5) 已看過的資料：o4 o_stagger_tranche 在 V3 資料上看過 J0／J+1 全期（官方 AB 對 J0 −0.06，t −0.38；對 J+1 +0.12，t 1.10），本題 2019 年後不是盲測，判定上限＝可用（定義）／預期依規則下修'],
 'must_report': ['逐 seed 分布（t 中位數、範圍、跌破 2.3／2.0 的 seed 數，拿掉最好 1 個月）', 'j ≥ 0（J0～J+12）各版本 AB t 的中位數與 p10', '端點敏感度（拿掉 2006、起點 2007／2008、拿掉最後 1 個月、最好 1／3／5／10 個月、最好 1／3 年、日曆月 t）',
                 '一般／壓力成本、損益兩平滑價', '與 0050、同池等權、主策略 AB_V4（o_v4 navs_v4.parquet）', '2024～2026', '30 萬與 1,000 萬帳戶每月下單次數', '沒有差異時附 MDE（80% 檢定力、雙尾 5%）',
                 '分年、分產業、去掉最大貢獻股', 'SOP 與 main_picks_today.py 的日期文字（訊號日與成交日分開）'],
 'annualization': '簡單報酬；年化從該版本第一個訊號日（2006-04 錨定月）收盤起算到最後一個評價日收盤（2026-10 錨定月的同型訊號日；J0／J0N＝2026-10-12、J+1＝2026-10-13）；分段年化從段內第一個訊號日收盤到段後第一個訊號日收盤',
 'periods': {'SEL': '2006-04～2018-12', 'VAL': '2019-01～2023-12', 'HOLD': '2024-01～2026-10', 'ALL': '2006-04～2026-10'},
}
canon = json.dumps(PLAN, ensure_ascii=False, sort_keys=True).encode()
f = HERE / 'plan.json'
assert not f.exists(), 'plan.json 已存在，不得覆寫'
f.write_text(json.dumps({'PLAN': PLAN, 'PLAN_sha256': hashlib.sha256(canon).hexdigest()}, ensure_ascii=False, indent=1), encoding='utf-8')
print(hashlib.sha256(canon).hexdigest(), PLAN['created_utc'])
