"""S93 格子（12 格，事前固定；看完訓練期描述 step0.json 後、任何含集保條件的模擬之前寫定）。F＝s93lib.TDCC.feat() 的特徵 dict。
門檻取訓練期現行交易分佈的約略第 10／90 百分位（dB4 p10 −1.54、dB8 p10 −2.64、dH4 p90 +13%、dR4 p90 +2.2）取整。"""
BAD = {       # 過濾：True → 不買
    'BD4': lambda f: f['dB4'] <= -1.5,                     # 近 4 週 400 張以上大戶持股比例下降 ≥ 1.5 個百分點
    'BD8': lambda f: f['dB8'] <= -2.5,                     # 近 8 週下降 ≥ 2.5 個百分點
    'HS4': lambda f: f['dH4'] >= 0.10,                     # 近 4 週股東人數增加 ≥ 10%（籌碼發散）
    'RU4': lambda f: f['dR4'] >= 2.0,                      # 近 4 週 10 張以下散戶持股比例增加 ≥ 2 個百分點
}
GOOD = {      # 確認：True → 才買（缺值照買）
    'CONC4': lambda f: (f['dB4'] > 0) & (f['dH4'] < 0),    # 近 4 週大戶比例增加且股東人數減少（籌碼集中）
}
EXIT = {      # 出場：訊號日 True → 次一交易日開盤賣（只在訊號日檢查）
    'XD3H': lambda f: (f['ndec'] >= 3) & (f['dH4'] > 0),   # 大戶比例連續 3 筆（週）下降且近 4 週股東人數增加
    'XD4B': lambda f: (f['ndec'] >= 4) & (f['dB4'] <= -1.0),   # 大戶比例連續 4 筆（週）下降且近 4 週合計下降 ≥ 1 個百分點
}
CELLS = tuple([f'{r}|{m}' for r in BAD for m in 'ZF'] + [f'{r}|{m}' for r in GOOD for m in 'ZF'] + list(EXIT))
assert len(CELLS) == 12
