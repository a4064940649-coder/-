#!/bin/bash
# 逐檔下載 tw_stocker 原始 5 分 K → 彙總 → 刪原檔。用法：bash fetch_all.sh
S=/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad
D=/home/user/-/research/tw_stock_selection/agents/value_fix
mkdir -p $S/vf_raw $S/vf_agg
ls $S/tw_stocker_daily/ | sed 's/\.csv$//' > $S/vf_codes.txt
one() {
  c=$1; [ -s $S/vf_agg/$c.parquet ] && return
  for t in 1 2 3; do
    curl -sS --max-time 180 -o $S/vf_raw/$c.csv https://raw.githubusercontent.com/voidful/tw_stocker/main/data/$c.csv && break
    sleep 2
  done
  python3 -I -B $D/agg_raw5m.py $S/vf_raw/$c.csv $S/vf_agg/$c.parquet && echo "ok $c" || echo "fail $c"
  rm -f $S/vf_raw/$c.csv
}
export -f one; export S D
cat $S/vf_codes.txt | xargs -P 6 -I{} bash -c 'one {}'
