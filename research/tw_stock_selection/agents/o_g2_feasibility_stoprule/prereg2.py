#!/usr/bin/env python3
"""第二次登記：最終模擬前登記資料與生成器程式的 sha256（PREREG2）。不得覆寫。"""
import json, datetime, sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
import g2lib as G

r = G.check_prereg()
assert 'PREREG2' not in r, 'PREREG2 已存在'
files = ['g2lib.py', 's3_inputs.py', 's4_sim.py', 'data/sim_inputs.npz', 'data/family_ir.parquet', 'data/fund_ir.parquet', 'prereg.py']
P2 = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
      'hashes': {f: G.sha(G.HERE / f) for f in files},
      'notes': ['生成器依 PREREG 實作：V1 主變體（L=6、σ＝AB_V4 全期月超額 sd＝5.99pp、ε＝AB_V4 標準化平穩 bootstrap），V2～V7 依 PREREG 的 variants_iterations。',
                '實作細節（PREREG 未寫死、在此登記）：G1 的 p10 用 np.percentile 線性內插；DSR 的 var_SR 每次把候選 SR 併入族群 2,334 條後重算；前瞻延長中 SPA 用族群最大值分布近似（候選自己的統計量略去）；C 的 Pocock 邊界由 200,000 條 iid 常態路徑的 95% 分位數求得；C 的 G7 各級只要求全期 t≥3。',
                '揭露：登記前曾在 scratchpad 以 NSIM=60、IR=0.5 做冒煙測試（只為除錯；輸出未使用），看到 G7、G2 的粗略通過率；生成器與門檻未因此改動，只修了一個抽樣數上限的程式錯誤。',
                '主策略後驗點（AB_V4 後驗平均、V7X 後驗平均、AB_V4 量測值）為描述用額外格點，非事前登記的 IR 格點。']}
r['PREREG2'] = P2; r['PREREG2_sha256'] = G.canon(P2)
G.RES.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print('PREREG2', r['PREREG2_sha256'], P2['created_utc'])
