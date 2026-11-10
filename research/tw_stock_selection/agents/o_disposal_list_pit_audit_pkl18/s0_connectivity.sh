#!/bin/bash
# 每台主機只測一次（不重試，不改環境）；結果寫入 raw/step0_connectivity.txt
cd "$(dirname "$0")"
OUT=raw/step0_connectivity.txt
echo "# step0 connectivity  utc=$(date -u +%FT%TZ)  (single attempt per host, curl --max-time 20, via HTTPS_PROXY)" > $OUT
for u in "https://mops.twse.com.tw/mops/web/t05st01" "https://mopsov.twse.com.tw/mops/web/t05st01" "https://www.twse.com.tw/zh/announcement/punish.html" "https://www.twse.com.tw/rwd/zh/announcement/punish?response=json" "https://web.archive.org/web/2019/https://www.twse.com.tw/zh/page/announcement/punish.html" "https://www.tpex.org.tw/" "https://raw.githubusercontent.com/tsungmin2001/tw-disposition/main/README.md" "https://api.github.com/repos/tsungmin2001/tw-disposition" "https://law.moj.gov.tw/" "https://twse-regulation.twse.com.tw/" ; do
  r=$(curl -sS -o /dev/null -w '%{http_code} %{time_total}' --max-time 20 "$u" 2>&1 | tr '\n' ' ')
  echo "$u -> $r" >> $OUT
done
cat $OUT
