#!/bin/bash
# 備援（基礎建設判不可用或當天產生器失敗時）：手動跑 main_picks_today.py，輸出加時間戳與 sha256 存檔。
# 用法：bash fallback_manual.sh [資料目錄，預設本題 raw/screener/data]
set -u
F=/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live
TW=/home/user/-/research/tw_stock_selection
D=${1:-$F/raw/screener/data}
OUT=/home/user/-/research/tw_stock_selection/agents/live_tracking/main_forward/fallback
mkdir -p $OUT
TS=$(TZ=Asia/Taipei date +%Y%m%dT%H%M%S%z)
f=$OUT/main_picks_${TS}.txt
{
  echo "# wall_clock_taipei=$(TZ=Asia/Taipei date -Iseconds)"
  echo "# data_head=$(git -C $D/.. log -1 --format='%H %cI' 2>/dev/null)"
  echo "# main_picks_today.sha256=$(sha256sum $TW/main_picks_today.py | cut -d' ' -f1)"
  python3 -I $TW/main_picks_today.py --data $D --n 10 --ex-fin
} > $f 2>&1
sha256sum $f >> $OUT/SHA256SUMS
python3 -I $TW/disposal_today.py --data $D --taiex $F/raw/finlab/taiex_twii.csv --tpex > $OUT/disposal_${TS}.txt 2>&1
sha256sum $OUT/disposal_${TS}.txt >> $OUT/SHA256SUMS
echo "寫入 $f（sha256 已記錄於 $OUT/SHA256SUMS）"
