#!/usr/bin/env python3
"""唯讀快照：live_tracking/main_forward 與 o_forward_infra_live 逐檔 sha256＋mtime（UTC）。
清單 sha＝sha256sum 格式（"<sha>  <relpath>\\n"，依路徑排序）的 sha256，與 G8P-1 READONLY_CHECK 同法（已驗證可重現 639dd8b0…）。
用法：python3 -I -B ro_snapshot.py start|end
start/end 各寫 ro_<tag>.json；end 時另與 start、G8P-1 基準比較，並把差異檔的 mtime 對到每日排程日誌的執行區間。"""
import sys, json, hashlib, datetime, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
AG = HERE.parent
FOLDERS = ['live_tracking/main_forward', 'o_forward_infra_live']


def snap():
    files = {}
    for f in FOLDERS:
        for p in sorted((AG / f).rglob('*')):
            if p.is_file():
                rel = str(p.relative_to(AG))
                files[rel] = dict(sha=hashlib.sha256(p.read_bytes()).hexdigest(),
                                  mtime=datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).isoformat())
    lst = ''.join(f"{v['sha']}  {k}\n" for k, v in sorted(files.items()))
    return files, hashlib.sha256(lst.encode()).hexdigest()


def sched_windows():
    """每日排程日誌：檔名 daily_YYYYMMDD_HHMMSS.log（台北時間）＝開始；以檔案 mtime 為結束。"""
    out = []
    for p in sorted((AG / 'o_forward_infra_live/logs').glob('daily_*.log')):
        m = re.match(r'daily_(\d{8})_(\d{6})\.log', p.name)
        if not m: continue
        st = datetime.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
        en = datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc)
        out.append((p.name, st.astimezone(datetime.timezone.utc), en))
    return out


def main(tag):
    files, lsha = snap()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    out = dict(tag=tag, utc=now, n_files=len(files), listing_sha256=lsha, files=files)
    if tag == 'end':
        st = json.loads((HERE / 'ro_start.json').read_text(encoding='utf-8'))
        t0 = datetime.datetime.fromisoformat(st['utc'])
        changed = sorted(k for k in set(files) | set(st['files']) if files.get(k, {}).get('sha') != st['files'].get(k, {}).get('sha'))
        W = sched_windows()
        attrib = {}
        for k in changed:
            mt = files.get(k, {}).get('mtime')
            if mt is None: attrib[k] = 'DELETED（非本題可能來源，須調查）'; continue
            mt = datetime.datetime.fromisoformat(mt)
            hit = [w[0] for w in W if w[1] - datetime.timedelta(seconds=5) <= mt <= w[2] + datetime.timedelta(seconds=5)]
            attrib[k] = hit[0] if hit else 'UNATTRIBUTED'
        out.update(start_listing_sha256=st['listing_sha256'], start_utc=st['utc'], identical=(lsha == st['listing_sha256']),
                   changed=changed, changed_attribution=attrib,
                   n_unattributed=sum(v.startswith('UNATTRIBUTED') or v.startswith('DELETED') for v in attrib.values()),
                   sched_runs_during_topic=[w[0] for w in W if w[1] >= t0 - datetime.timedelta(seconds=5)])
    (HERE / f'ro_{tag}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'files'}, ensure_ascii=False, indent=1)[:4000])


if __name__ == '__main__':
    main(sys.argv[1])
