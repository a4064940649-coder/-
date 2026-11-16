#!/bin/bash
# S26 依優先序逐一（單一行程、依序）在修正開啟下重跑各題腳本。用法：bash batch.sh <job...>，job＝資料夾:腳本[:--no-frozen]
L=/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/s26/logs; mkdir -p $L
cd "$(dirname "$0")"
for job in "$@"; do
  IFS=: read d s x <<< "$job"
  tag="${d%%_*}_${s%.py}_fix1${x:+_nf}"
  echo "=== $(date +%T) $tag"
  timeout 5400 python3 -I -B runx.py "$d" "$s" 1 "$tag" $x > "$L/$tag.log" 2>&1
  echo "rc=$? $(tail -1 $L/$tag.log)"
done
