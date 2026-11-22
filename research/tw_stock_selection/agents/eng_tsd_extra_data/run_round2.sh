#!/bin/bash
# 用法：run.sh <資料集名,逗號分隔> ；臨時 clone 在 $W/c（沒有就開），做完由呼叫者刪
set -uo pipefail
W=/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/r2
LOG=/home/user/-/research/tw_stock_selection/agents/eng_tsd_extra_data/disk_log.tsv
avail(){ df -m --output=avail /home/user | tail -1 | tr -d ' '; }
A0=$(avail); if [ "$A0" -lt 1200 ]; then echo "可用 ${A0}MB < 1200MB 停"; exit 3; fi
[ -d $W/c ] || { git clone -q --filter=blob:none --depth=1 --no-checkout https://github.com/jb7k6295r7-bot/tw-stock-data $W/c; }
cd $W
( while sleep 20; do a=$(avail); [ "$a" -lt 1200 ] && { echo "磁碟低於 1200MB，中止"; pkill -f "update_daily.py $W/c"; break; }; done ) &
WD=$!
python3 -I -B $W/rss.py python3 -I -B /home/user/-/research/tw_stock_selection/raw/tsd_extra/update_daily.py $W/c --out $W/stage --only "$1" > $W/log_$2.txt 2>&1; RC=$?
kill $WD 2>/dev/null
A1=$(avail); CS=$(du -sm $W/c | cut -f1)
printf '%s\tround2:%s\tbefore=%sMB\tclone=%sMB\tafter_build=%sMB\trc=%s\tmaxrss_kb=%s\n' "$(date -Is)" "$1" "$A0" "$CS" "$A1" "$RC" "$(grep 'Maximum resident' $W/log_$2.txt | awk '{print $NF}')" | tee -a $LOG
grep -v '^\s' $W/log_$2.txt | grep -E '已更新|失敗|結果|無新資料' | tail -30
