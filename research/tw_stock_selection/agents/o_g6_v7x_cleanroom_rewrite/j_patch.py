# J 角色診斷用（不屬凍結程式）：在 World 物件上替換 V60／vrank 的口徑，找下一個分歧。
# mode 以逗號分隔：v60strict（GOOD 日窗內缺值→NaN，只有非 GOOD 日沿用前一日）、v60lim3（原引擎 ffill(limit=3)）、vrcf（vrank 收盤條件用前值補收盤 CF）
import numpy as np, pandas as pd
def patch(W, mode):
    if not mode: return W
    ms = set(mode.split(','))
    X = W.X; V = X['V']; GOOD = X['GOOD']; dates = W.dates; T, N = V.shape
    gi = np.where(GOOD)[0]
    EX = [pd.Timestamp('2017-08-03'), pd.Timestamp('2018-07-12')] + list(pd.date_range('2024-02-15', '2024-03-15'))
    ex = np.isin(dates[gi], pd.DatetimeIndex(EX))
    Vg = V[gi]; nanv = np.isnan(Vg); bad = nanv & (~ex[:, None])
    s = pd.DataFrame(np.where(nanv, 0, Vg)).rolling(60, min_periods=60).sum().values
    n = pd.DataFrame((~nanv).astype(float)).rolling(60, min_periods=60).sum().values
    b = pd.DataFrame(bad.astype(float)).rolling(60, min_periods=60).sum().values
    with np.errstate(invalid='ignore', divide='ignore'):
        v60g = np.where((b == 0) & (n > 0), s / n, np.nan)
    V60 = np.full((T, N), np.nan); V60[gi] = v60g
    if 'v60lim3' in ms:
        V60 = pd.DataFrame(V60).ffill(limit=3).values
    elif 'v60strict' in ms:
        for t in np.where(~GOOD)[0]:
            if t > 0: V60[t] = V60[t - 1]
    else:
        V60 = pd.DataFrame(V60).ffill().values
    CL = X['CF'] if 'vrcf' in ms else X['C']
    elig = (V60 >= 1e7) & (CL >= 10)
    key = np.where(elig, V60, -np.inf)
    order = np.argsort(-key, axis=1, kind='stable')
    rows = np.arange(T)[:, None]
    rk = np.empty_like(order); rk[rows, order] = np.arange(1, N + 1)[None, :]
    VR = np.where(elig, rk, np.nan).astype(float)
    sh = lambda a: np.vstack([np.full((1, a.shape[1]), np.nan), a[:-1]])
    W.V60p = sh(V60); W.VRp = sh(VR)
    vr = W.VRp
    W.DEC = np.clip(np.where(np.isnan(vr) | (vr > 300), 10, np.ceil(np.nan_to_num(vr, nan=3000) / 30)).astype(int), 1, 10)
    W.in300 = ~(np.isnan(vr) | (vr > 300))
    return W

def patch_tiecol():
    """診斷：換股日同 delta 時依欄位序（原引擎 np.union1d＋stable argsort）"""
    import inspect, textwrap, xengine
    src = textwrap.dedent(inspect.getsource(xengine.Sim.rebalance))
    assert 'sorted(ds.items(), key=lambda kv: kv[1])' in src
    src = src.replace('sorted(ds.items(), key=lambda kv: kv[1])', 'sorted(ds.items(), key=lambda kv: (kv[1], kv[0]))')
    ns = {}; exec(src, xengine.__dict__, ns); xengine.Sim.rebalance = ns['rebalance']

def patch_oddlots():
    """診斷：順延日零股判定的「張數」用未套參與上限的 min(待買張數, 現金張數上限)（原引擎讀法）"""
    import inspect, textwrap, xengine
    src = textwrap.dedent(inspect.getsource(xengine.Sim.exec_order))
    a = "                lots_x = min(lots_x, int(math.floor(self.cash / (o * 1000 * 1.05))))\n                if odd_x > 0:\n                    oc_cash, _ = self.leg_cost(t, j, 1, 0, odd_x)\n                    if -oc_cash > self.cash - lots_x * o * 1050: odd_x = 0\n"
    a = textwrap.dedent(inspect.getsource(xengine.Sim.exec_order)) and a.replace('                ', '            ', 1)
    src0 = src
    old = "            lots_x = min(lots_x, int(math.floor(self.cash / (o * 1000 * 1.05))))\n"
    assert old in src, 'anchor'
    src = src.replace(old, "            Lchk = min(lots, int(math.floor(self.cash / (o * 1000 * 1.05))))\n            lots_x = min(lots_x, Lchk)\n")
    src = src.replace("                if -oc_cash > self.cash - lots_x * o * 1050: odd_x = 0\n", "                if -oc_cash > self.cash - Lchk * o * 1050: odd_x = 0\n")
    assert src != src0
    ns = {}; exec(src, xengine.__dict__, ns); xengine.Sim.exec_order = ns['exec_order']
