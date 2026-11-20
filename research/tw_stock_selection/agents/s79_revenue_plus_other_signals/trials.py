#!/usr/bin/env python3
"""S79 搜尋空間（第一階段）。每個試驗＝dict(id, fam, sub, grid, spec)；spec 的意義見 s79lib.make_signal。
fam＝附加資訊的種類；sub＝同一條參數軸（或格子）的小組；grid＝在小組內的座標（鄰格＝同 sub、座標只有一維差 1）。"""
DEF_PG = [['RET20', '>', 0.10]]


def build():
    T = []
    def add(fam, sub, grid, tag, **spec):
        T.append(dict(id=f'{fam}.{sub}.{tag}', fam=fam, sub=f'{fam}.{sub}', grid=list(grid), spec=spec))
    add('F0', 'base', (0,), 'current')
    # F1 價格動能（3／6／12 個月報酬；作用在前 10 名的「已在漲」條件）
    for i, n in enumerate((60, 120, 240)):
        for j, x in enumerate((0.0, 0.1, 0.2, 0.3)):
            add('F1', 'repl', (i, j), f'RET{n}>{x}', pool_gates=[[f'RET{n}', '>', x]])
        for j, x in enumerate((0.0, 0.1, 0.2)):
            add('F1', 'and', (i, j), f'RET20&RET{n}>{x}', pool_gates=DEF_PG + [[f'RET{n}', '>', x]])
    for j, x in enumerate((0.0, 0.2)):
        add('F1', 'm12_1', (j,), f'MOM12_1>{x}', pool_gates=[['MOM12_1', '>', x]])
    # F2 距 52 週高點
    for j, x in enumerate((0.7, 0.8, 0.85, 0.9, 0.95)):
        add('F2', 'list', (j,), f'NH52>={x}', list_gates=[['NH52', '>=', x]])
        add('F2', 'pool', (j,), f'poolNH52>={x}', pool_gates=[['NH52', '>=', x]])
    # F3 低波動（股池內百分位）
    for j, x in enumerate((0.5, 0.6, 0.7, 0.8, 0.9)):
        add('F3', 'lo60', (j,), f'VOL60pct<={x}', list_gates=[['VOL60_PCT', '<=', x]])
    for j, x in enumerate((0.3, 0.5)):
        add('F3', 'hi60', (j,), f'VOL60pct>={x}', list_gates=[['VOL60_PCT', '>=', x]])
    for j, x in enumerate((0.5, 0.7, 0.9)):
        add('F3', 'lo120', (j,), f'VOL120pct<={x}', list_gates=[['VOL120_PCT', '<=', x]])
    # F4 成交值／流動性
    for j, x in enumerate((2e7, 3e7, 5e7, 1e8, 2e8)):
        add('F4', 'min', (j,), f'V60>={int(x / 1e4)}萬', list_gates=[['V60', '>=', x]])
    for j, x in enumerate((1e8, 3e8, 1e9)):
        add('F4', 'max', (j,), f'V60<={int(x / 1e4)}萬', list_gates=[['V60', '<=', x]])
    for j, x in enumerate((1.0, 1.5)):
        add('F4', 'vratio', (j,), f'VRATIO>={x}', list_gates=[['VRATIO', '>=', x]])
    # F5 股價位階
    for j, x in enumerate((15, 20, 30, 50)):
        add('F5', 'min', (j,), f'PX>={x}', list_gates=[['PX', '>=', x]])
    for j, x in enumerate((50, 100, 200)):
        add('F5', 'max', (j,), f'PX<={x}', list_gates=[['PX', '<=', x]])
    # F6 營收年增率與 SURGE 合成
    for j, x in enumerate((0.0, 0.1, 0.2, 0.3)):
        add('F6', 'yoy1', (j,), f'YOY1>{x}', list_gates=[['YOY1', '>', x]])
    for j, x in enumerate((0.0, 0.1, 0.2)):
        add('F6', 'yoy3', (j,), f'YOY3>{x}', list_gates=[['YOY3', '>', x]])
    for i, v in enumerate(('YOY1', 'YOY3')):
        for j, w in enumerate((0.75, 0.5, 0.25)):
            add('F6', 'blend', (i, j), f'{w}S+{v}', score=dict(type='blend', x=v, w=w))
    # F7 營收連續成長
    for j, x in enumerate((2, 3, 6, 12)):
        add('F7', 'consec', (j,), f'CONSEC>={x}', list_gates=[['CONSEC', '>=', x]])
    for j, x in enumerate((8, 10)):
        add('F7', 'pos12', (j,), f'POS12>={x}', list_gates=[['POS12', '>=', x]])
    add('F7', 'revhi', (0,), 'REVHI12', list_gates=[['REVHI12', '>=', 1]])
    for j, x in enumerate((0.0, 1.0)):
        add('F7', 'sprev', (j,), f'SURGE_PREV>{x}', list_gates=[['SURGE_PREV', '>', x]])
    # F8 產業內相對（分類檔是現在的分類＝有前視與存活者疑慮）
    add('F8', 'demean', (0,), 'ind_demean', score=dict(type='ind_demean'))
    add('F8', 'irank', (0,), 'ind_rank', score=dict(type='ind_rank'))
    for j, x in enumerate((0.0, 0.5)):
        add('F8', 'imed', (j,), f'IND_MED>{x}', list_gates=[['IND_MED', '>', x]])
    for j, x in enumerate((1, 2, 3)):
        add('F8', 'itop', (j,), f'IND_TOP{x}', list_gates=[['IND_TOPN', '<=', x]])
    for j, x in enumerate((2, 3)):
        add('F8', 'icap', (j,), f'IND_CAP{x}', ind_cap=x)
    # F9 SURGE 與價量特徵的合成排名
    for i, (v, sgn) in enumerate((('RET60', 1), ('RET120', 1), ('RET240', 1), ('NH52', 1), ('VOL60', -1), ('V60', -1), ('V60', 1))):
        for j, w in enumerate((0.75, 0.5)):
            add('F9', f'{v}{"-" if sgn < 0 else "+"}', (j,), f'{w}S', score=dict(type='blend', x=v, w=w, sign=sgn))
    # F11 過熱迴避
    for j, x in enumerate((0.3, 0.5)):
        add('F11', 'ret20cap', (j,), f'RET20<{x}', pool_gates=DEF_PG + [['RET20', '<', x]])
    for j, x in enumerate((0.2, 0.3)):
        add('F11', 'dist60', (j,), f'DIST60<={x}', list_gates=[['DIST60', '<=', x]])
    assert len({t['id'] for t in T}) == len(T)
    return T


def neighbors(T):
    out = {}
    for a in T:
        nb = [b['id'] for b in T if b['sub'] == a['sub'] and b['id'] != a['id'] and len(b['grid']) == len(a['grid'])
              and sum(abs(x - y) for x, y in zip(a['grid'], b['grid'])) == 1]
        out[a['id']] = nb
    return out


if __name__ == '__main__':
    T = build(); nb = neighbors(T)
    from collections import Counter
    print(len(T), Counter(t['fam'] for t in T), sum(1 for v in nb.values() if not v))
