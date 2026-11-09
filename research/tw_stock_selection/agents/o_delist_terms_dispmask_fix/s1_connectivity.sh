#!/bin/bash
# 每台主機只測一次（不重試）；結果寫入 raw/step0_connectivity.txt
OUT=raw/step0_connectivity.txt
echo "# step0 connectivity  utc=$(date -u +%FT%TZ)  (single attempt per host, curl --max-time 20, via HTTPS_PROXY)" > $OUT
for u in "https://mops.twse.com.tw/mops/web/t05st01" "https://mopsov.twse.com.tw/mops/web/t05st01" "https://emops.twse.com.tw/" "https://www.twse.com.tw/zh/announcement/delisting.html" "https://web.archive.org/web/2010/http://mops.twse.com.tw/" "https://zh.wikipedia.org/wiki/%E5%80%9A%E5%A4%A9%E8%B3%87%E8%A8%8A" "https://news.cnyes.com/" "https://www.cna.com.tw/" "https://money.udn.com/" "https://api.github.com/repos/aRthur08701244/Intern-in-CTBC" "https://raw.githubusercontent.com/ji08kk44/BDSE26_TEAM3/13882587/README.md" "https://github.com/nickchengtw/stock-llm-neuralprophet"; do
  r=$(curl -sS -o /dev/null -w '%{http_code} %{time_total}' --max-time 20 "$u" 2>&1 | tr '\n' ' ')
  echo "$u -> $r" >> $OUT
done
cat $OUT
