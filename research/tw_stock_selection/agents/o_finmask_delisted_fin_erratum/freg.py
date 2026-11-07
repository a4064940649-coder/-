#!/usr/bin/env python3
"""本題共用的輕量工具：sha256、UTC、登記（寫檔→算雜湊→記入 results.json）。不 import 任何引擎。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o16_finmask')
RES = HERE / 'results.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def now_utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')

def load_res():
    return json.loads(RES.read_text(encoding='utf-8')) if RES.exists() else {}

def save_res(r):
    RES.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

def register(name, obj, fname):
    """把 obj 寫成 fname（本資料夾），算 sha256，記入 results.json[name]。已登記者不得覆寫。"""
    p = HERE / fname
    r = load_res()
    assert name not in r, f'{name} 已登記，不得覆寫'
    assert not p.exists(), f'{fname} 已存在，不得覆寫'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    h = sha(p)
    r[name] = dict(file=fname, sha256=h, utc=obj.get('utc'))
    save_res(r)
    return h

def check_reg(name):
    r = load_res()
    assert name in r, f'{name} 尚未登記'
    p = HERE / r[name]['file']
    assert p.exists() and sha(p) == r[name]['sha256'], f'{name} 雜湊不符'
    return json.loads(p.read_text(encoding='utf-8')), r[name]
