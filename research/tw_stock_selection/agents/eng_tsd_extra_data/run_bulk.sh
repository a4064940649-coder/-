#!/bin/bash
# 首次建置用：一次一個資料集——開臨時 clone → update_daily.py --only <資料集> → 刪臨時 clone。前後記錄磁碟；可用空間低於 800MB 不做。
# 用法：run_bulk.sh <臨時工作資料夾> <輸出資料夾> <資料集名稱…>
set -uo pipefail
W="$1"; OUT="$2"; shift 2
H="$(cd "$(dirname "$0")" && pwd)"; LOG="$H/disk_log.tsv"; U="$OUT/update_daily.py"
avail(){ df -m --output=avail /home/user | tail -1 | tr -d ' '; }
for DS in "$@"; do
  A0=$(avail); if [ "$A0" -lt 800 ]; then echo "可用空間 ${A0}MB < 800MB，停止（$DS 未做）"; exit 3; fi
  rm -rf "$W/c"; mkdir -p "$W"
  git clone -q --filter=blob:none --depth=1 --no-checkout https://github.com/jb7k6295r7-bot/tw-stock-data "$W/c" || { echo "clone 失敗"; exit 4; }
  (cd /tmp && python3 -I -B "$U" "$W/c" --out "$OUT" --only "$DS" --quiet); RC=$?
  A1=$(avail); CS=$(du -sm "$W/c" | cut -f1); rm -rf "$W/c"; A2=$(avail)
  printf '%s\t%s\trc=%s\tbefore=%sMB\tpeak_clone=%sMB\tafter_build=%sMB\tafter_cleanup=%sMB\tout=%sMB\n' "$(date -Is)" "$DS" "$RC" "$A0" "$CS" "$A1" "$A2" "$(du -sm "$OUT/$DS" | cut -f1)" | tee -a "$LOG"
done
