#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_smc_taiwan.py
======================

Mechanical, daily-bar proxies of "Smart Money Concepts" (SMC) / ICT ideas and
classic support–resistance rules, backtested on Taiwan equities / TAIEX with a
realistic Taiwan transaction-cost model.

Design goals
------------
* One file, pandas + numpy only (yfinance is optional and only needed for the
  default data source).  Python 3.10+, pandas 2.x / 3.x.
* Every rule is a short, self-contained function (``rule_r1_sweep`` ...).
* Every trade is an explicit record (signal/entry/exit index & date, prices,
  stop, target, exit reason) written to CSV so it can be audited.
* All statistics are computed from those trade records; nothing is hidden.
* Deterministic: a single ``--seed`` drives the bootstrap, the shuffled-bar
  surrogates and the synthetic smoke-test data.

Timing conventions (no look-ahead)
----------------------------------
* A signal on bar ``t`` only uses bars ``<= t``.
* ``entry_kind='close'``  (R1 default): enter at the close of the signal bar.
  This assumes the order is placed in the 13:25–13:30 closing auction using the
  pre-close indicative price – a mild optimism.  ``--r1-entry next_open`` removes
  it (entry at the next bar's open).
* ``entry_kind='open'``   (R4a, R4b, R1 with ``--r1-entry next_open``): enter at
  the open of bar ``t+1``.
* ``entry_kind='limit'``  (R2, R3): a resting limit order at a price level is
  filled on the first later bar whose range touches the level.  Fill price is
  the level, or the open if the bar opens through the level (better fill).  If
  the bar opens *beyond the stop* the order is treated as cancelled (the zone
  was invalidated before we could trade it).
* Exits: stop / target are checked bar by bar.  If one bar's range covers both
  stop and target the stop is assumed to be hit first (conservative).  Gaps
  through a level are filled at the open.  On a limit-entry bar only the stop can
  trigger (the high may have printed before the fill).  Time exits are at the
  close of the N-th bar held.

Cost model (Taiwan, user should verify current rates)
------------------------------------------------------
* Cash stock, overnight: commission 0.1425 % x discount per side (default 6 折
  = 0.0855 %), securities transaction tax 0.30 % on the sell, optional margin
  short borrow fee for shorts.
* Cash stock, same-day (現股當沖): tax 0.15 % instead of 0.30 %.  A daily-bar
  strategy holds overnight, so this is reported as a *hypothetical* lower-cost
  scenario only.
* Futures: tax 0.002 % of notional + NT$ commission per contract per side,
  notional = price x multiplier (小台 MTX: 50 NT$/pt for the index; 2 000 shares
  for single-stock futures).  Margin, basis and roll costs are ignored.
* Slippage: ``--slippage-bps`` per side (default 5 bp) or ``--slippage-ticks``
  using Taiwan tick-size bands.

Nothing in this file downloads data unless the user runs it with a network
connection.  ``--smoke-test`` runs the whole pipeline on in-memory synthetic
random walks – the output of that mode is NOT a real backtest result.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

__version__ = "1.0.0"

LOG = logging.getLogger("smc_tw")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEFAULT_TICKERS: list[str] = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW", "2412.TW",
    "2881.TW", "2882.TW", "3711.TW", "2303.TW", "0050.TW", "^TWII",
]

RULE_R1 = "R1_sweep"
RULE_R2 = "R2_fvg"
RULE_R3 = "R3_orderblock"
RULE_R4A = "R4a_sr_bounce"
RULE_R4B = "R4b_breakout"
RULE_ORDER = [RULE_R1, RULE_R2, RULE_R3, RULE_R4A, RULE_R4B]

RULE_DESCRIPTIONS: dict[str, str] = {
    RULE_R1: "Liquidity sweep reversal (ICT turtle soup / PDL-PDH sweep): low < prior low and close back above it (mirror for shorts). "
             "Entry at signal close (default) or next open; stop beyond the sweep extreme; target k R; time exit after --hold bars.",
    RULE_R2: "Fair value gap (3-candle imbalance): bullish when high[t-2] < low[t]. Resting limit order at the gap's near edge, "
             "stop beyond the far edge minus buffer, target k R, time exit after --level-hold bars.",
    RULE_R3: "Order block retest: last opposite-colour candle before a new 20-bar high/low. Limit order at the OB's near edge, "
             "stop beyond the OB's far edge, target k R, time exit after --level-hold bars.",
    RULE_R4A: "Classic S/R bounce: close within tol% of the rolling 20-bar low/high, next bar reverses, enter next open, "
              "stop beyond the level, target k R, time exit after --level-hold bars.",
    RULE_R4B: "Classic breakout: close beyond the prior 20-bar high/low, enter next open, hold --hold bars (no stop; R unit = ATR14).",
}

COST_MODES: tuple[str, ...] = ("gross", "cash_overnight", "cash_daytrade", "futures")
RET_COL = {"gross": "ret_gross", "cash_overnight": "ret_net_cash_overnight",
           "cash_daytrade": "ret_net_cash_daytrade", "futures": "ret_net_futures"}
R_COL = {"gross": "R_gross", "cash_overnight": "R_net_cash_overnight",
         "cash_daytrade": "R_net_cash_daytrade", "futures": "R_net_futures"}

FVG_HORIZONS: tuple[int, ...] = (5, 10, 20)

Side = Literal["long", "short"]
EntryKind = Literal["close", "open", "limit"]
InstrumentKind = Literal["stock", "etf", "index"]


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class Config:
    """All run-time parameters (filled from argparse)."""

    tickers: list[str]
    source: str = "yfinance"
    start: str = "2015-01-01"
    end: str = field(default_factory=lambda: date.today().isoformat())
    data_dir: Path = Path("./data")
    out_dir: Path = Path("./output")
    csv_dir: Path = Path("./data_csv")
    finmind_token: str | None = None
    finmind_index_id: str = "TAIEX"
    refresh: bool = False
    adjust: bool = True
    # rule parameters
    sides: list[str] = field(default_factory=lambda: ["long", "short"])
    hold: int = 3
    level_hold: int = 10
    target_r: float = 2.0
    stop_buffer_bps: float = 10.0
    min_risk_pct: float = 0.2
    lookback: int = 20
    sr_tol_pct: float = 0.5
    fvg_wait: int = 20
    fvg_min_gap_pct: float = 0.0
    ob_lookback: int = 10
    ob_wait: int = 20
    r1_entry: str = "close"
    r1_strict: bool = False
    allow_overlap: bool = False
    # costs
    instrument: str = "cash"            # headline net column: cash -> cash_overnight, futures -> futures
    comm_rate: float = 0.001425
    comm_discount: float = 0.6
    tax_rate: float = 0.003
    daytrade_tax_rate: float = 0.0015
    short_borrow_bps: float = 8.0
    fut_tax_rate: float = 0.00002
    fut_comm: float = 50.0
    fut_multiplier: float | None = None  # None = auto (50 index / 2000 stock)
    slippage_bps: float = 5.0
    slippage_ticks: float = 0.0
    risk_per_trade: float = 0.01
    # statistics
    n_boot: int = 1000
    n_shuffle: int = 200
    seed: int = 42
    quick: bool = False
    smoke_test: bool = False

    @property
    def headline_mode(self) -> str:
        return "futures" if self.instrument == "futures" else "cash_overnight"

    @property
    def stop_buffer(self) -> float:
        return self.stop_buffer_bps / 1e4

    def to_jsonable(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


# ----------------------------------------------------------------------------
# Instrument helpers
# ----------------------------------------------------------------------------
INDEX_ALIASES = {"^TWII", "TAIEX", "TWII", "^TWOII", "TPEX"}


def instrument_kind(ticker: str) -> InstrumentKind:
    """Classify a ticker: index (^TWII, synthetic '^...'), ETF (00xx.TW) or stock."""
    t = ticker.upper()
    if t.startswith("^") or t in INDEX_ALIASES:
        return "index"
    code = t.split(".")[0]
    if code.startswith("00") and code[2:].isdigit():
        return "etf"
    return "stock"


def safe_name(ticker: str) -> str:
    """File-system safe name for a ticker ('^TWII' -> 'IDX_TWII')."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", ticker.replace("^", "IDX_"))


def tw_tick_size(price: float, kind: InstrumentKind) -> float:
    """Taiwan tick-size bands (TWSE stocks / ETFs; index futures tick = 1 point)."""
    if kind == "index":
        return 1.0
    if kind == "etf":
        return 0.01 if price < 50 else 0.05
    if price < 10:
        return 0.01
    if price < 50:
        return 0.05
    if price < 100:
        return 0.1
    if price < 500:
        return 0.5
    if price < 1000:
        return 1.0
    return 5.0


# ----------------------------------------------------------------------------
# Cost model
# ----------------------------------------------------------------------------
@dataclass
class CostModel:
    """Taiwan transaction costs. All rates are fractions (0.001425 = 0.1425 %)."""

    comm_rate: float = 0.001425
    comm_discount: float = 0.6
    tax_rate: float = 0.003
    daytrade_tax_rate: float = 0.0015
    short_borrow: float = 0.0008
    fut_tax_rate: float = 0.00002
    fut_comm: float = 50.0
    fut_multiplier: float | None = None
    slippage_bps: float = 5.0
    slippage_ticks: float = 0.0

    @classmethod
    def from_config(cls, cfg: Config) -> "CostModel":
        return cls(comm_rate=cfg.comm_rate, comm_discount=cfg.comm_discount, tax_rate=cfg.tax_rate,
                   daytrade_tax_rate=cfg.daytrade_tax_rate, short_borrow=cfg.short_borrow_bps / 1e4,
                   fut_tax_rate=cfg.fut_tax_rate, fut_comm=cfg.fut_comm, fut_multiplier=cfg.fut_multiplier,
                   slippage_bps=cfg.slippage_bps, slippage_ticks=cfg.slippage_ticks)

    def multiplier(self, kind: InstrumentKind) -> float:
        if self.fut_multiplier is not None:
            return self.fut_multiplier
        return 50.0 if kind == "index" else 2000.0

    def slippage_frac(self, price: float, kind: InstrumentKind) -> float:
        """Adverse slippage per side as a fraction of price."""
        if self.slippage_ticks > 0:
            return self.slippage_ticks * tw_tick_size(price, kind) / price
        return self.slippage_bps / 1e4

    def leg_costs(self, mode: str, price: float, kind: InstrumentKind) -> tuple[float, float]:
        """(buy-side cost, sell-side cost) as fractions of notional, excluding slippage."""
        if mode == "gross":
            return 0.0, 0.0
        if mode in ("cash_overnight", "cash_daytrade"):
            comm = self.comm_rate * self.comm_discount
            tax = self.tax_rate if mode == "cash_overnight" else self.daytrade_tax_rate
            return comm, comm + tax
        if mode == "futures":
            notional = price * self.multiplier(kind)
            per_side = self.fut_tax_rate + self.fut_comm / notional
            return per_side, per_side
        raise ValueError(f"unknown cost mode {mode}")

    def round_trip_pct(self, mode: str, price: float, kind: InstrumentKind) -> float:
        b, s = self.leg_costs(mode, price, kind)
        return 100.0 * (b + s + 2.0 * self.slippage_frac(price, kind))

    def net_return(self, side: str, entry_px: float, exit_px: float, mode: str, kind: InstrumentKind) -> float:
        """Net return on the entry notional after commissions, taxes, slippage (and borrow fee for cash shorts)."""
        slip_in = self.slippage_frac(entry_px, kind)
        slip_out = self.slippage_frac(exit_px, kind)
        if side == "long":
            buy_c, _ = self.leg_costs(mode, entry_px, kind)
            _, sell_c = self.leg_costs(mode, exit_px, kind)
            cost_in = entry_px * (1.0 + slip_in) * (1.0 + buy_c)
            proceeds = exit_px * (1.0 - slip_out) * (1.0 - sell_c)
            return proceeds / cost_in - 1.0
        # short: sell first (pays tax), buy to cover later
        _, sell_c = self.leg_costs(mode, entry_px, kind)
        buy_c, _ = self.leg_costs(mode, exit_px, kind)
        proceeds = entry_px * (1.0 - slip_in) * (1.0 - sell_c)
        cover = exit_px * (1.0 + slip_out) * (1.0 + buy_c)
        borrow = self.short_borrow if mode == "cash_overnight" else 0.0
        return (proceeds - cover) / entry_px - borrow


# ----------------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------------
_COLUMN_ALIASES = {
    "datetime": "date", "index": "date", "time": "date", "timestamp": "date", "日期": "date",
    "max": "high", "min": "low", "trading_volume": "volume", "vol": "volume", "volumn": "volume",
    "adjclose": "adj_close", "adj._close": "adj_close", "adjusted_close": "adj_close",
    "開盤價": "open", "最高價": "high", "最低價": "low", "收盤價": "close", "成交股數": "volume",
}


def standardize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise any OHLCV frame to a date-indexed frame with float columns
    open/high/low/close/volume[/adj_close], sorted and de-duplicated."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance >= 0.2.5x returns (field, ticker) columns for a single ticker
        df.columns = [str(c[0]) for c in df.columns]
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={c: _COLUMN_ALIASES.get(c, c) for c in df.columns})
    missing = [c for c in ("date", "open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; have {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = np.nan
    keep = ["date", "open", "high", "low", "close", "volume"] + (["adj_close"] if "adj_close" in df.columns else [])
    df = df[keep]
    dt = pd.to_datetime(df["date"], errors="coerce", utc=False)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    df = df.assign(date=dt.dt.normalize()).dropna(subset=["date"])
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").set_index("date")
    df.index.name = "date"
    return df


def apply_dividend_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Scale OHLC by adj_close/close so ex-dividend drops do not create fake gaps/sweeps."""
    if "adj_close" not in df.columns or df["adj_close"].isna().all():
        return df
    factor = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = df[c] * factor
    return out


def load_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is not installed: pip install yfinance (or use --source finmind/csv)") from exc
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    return standardize_ohlcv(raw)


def finmind_data_id(ticker: str, index_id: str) -> str:
    return index_id if instrument_kind(ticker) == "index" else ticker.split(".")[0]


def load_finmind(ticker: str, start: str, end: str, token: str | None, index_id: str) -> pd.DataFrame:
    """FinMind API v4, dataset TaiwanStockPrice (columns: date, open, max, min, close, Trading_Volume ...)."""
    params = {"dataset": "TaiwanStockPrice", "data_id": finmind_data_id(ticker, index_id),
              "start_date": start, "end_date": end}
    if token:
        params["token"] = token
    url = "https://api.finmindtrade.com/api/v4/data?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": f"smc-ict-taiwan-backtest/{__version__}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("status") not in (200, "200") or not payload.get("data"):
        raise RuntimeError(f"FinMind error for {ticker}: {payload.get('msg', payload)}")
    df = pd.DataFrame(payload["data"])
    return standardize_ohlcv(df)


def load_csv_source(ticker: str, csv_dir: Path) -> pd.DataFrame:
    for cand in (csv_dir / f"{safe_name(ticker)}.csv", csv_dir / f"{ticker}.csv"):
        if cand.exists():
            return standardize_ohlcv(pd.read_csv(cand))
    raise FileNotFoundError(f"no CSV for {ticker} in {csv_dir} (expected {safe_name(ticker)}.csv)")


def cache_paths(cfg: Config, ticker: str) -> tuple[Path, Path]:
    # NB: never use Path.with_suffix here – it would turn '2330.TW' into '2330.csv'.
    name = safe_name(ticker)
    return cfg.data_dir / f"{name}.csv", cfg.data_dir / f"{name}.meta.json"


def read_cache(cfg: Config, ticker: str) -> pd.DataFrame | None:
    csv_path, meta_path = cache_paths(cfg, ticker)
    if cfg.refresh or not csv_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("start", "9999") <= cfg.start and meta.get("end", "0000") >= cfg.end \
                and meta.get("source") == cfg.source:
            return standardize_ohlcv(pd.read_csv(csv_path))
    except Exception as exc:  # corrupt cache -> re-download
        LOG.warning("cache for %s unreadable (%s); re-downloading", ticker, exc)
    return None


def write_cache(cfg: Config, ticker: str, df: pd.DataFrame) -> None:
    csv_path, meta_path = cache_paths(cfg, ticker)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=True, date_format="%Y-%m-%d")
    meta = {"ticker": ticker, "source": cfg.source, "start": cfg.start, "end": cfg.end,
            "fetched_at": datetime.now().isoformat(timespec="seconds"), "rows": int(len(df))}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_instrument(cfg: Config, ticker: str) -> pd.DataFrame:
    """Load one instrument from cache or the configured source; cache the raw (unadjusted) frame."""
    df = read_cache(cfg, ticker)
    if df is None:
        LOG.info("downloading %s from %s (%s .. %s)", ticker, cfg.source, cfg.start, cfg.end)
        if cfg.source == "yfinance":
            df = load_yfinance(ticker, cfg.start, cfg.end)
        elif cfg.source == "finmind":
            df = load_finmind(ticker, cfg.start, cfg.end, cfg.finmind_token, cfg.finmind_index_id)
        elif cfg.source == "csv":
            df = load_csv_source(ticker, cfg.csv_dir)
        else:
            raise ValueError(f"unknown source {cfg.source}")
        if cfg.source != "csv":
            write_cache(cfg, ticker, df)
    else:
        LOG.info("using cached %s (%d rows)", ticker, len(df))
    df = df.loc[(df.index >= pd.Timestamp(cfg.start)) & (df.index <= pd.Timestamp(cfg.end))]
    if cfg.adjust:
        df = apply_dividend_adjustment(df)
    return df


MAX_GAP_DAYS = 10  # calendar days between consecutive bars above which a gap is reported (Lunar New Year ~ 9-10)


def validate_ohlcv(df: pd.DataFrame, ticker: str, max_gap_days: int = MAX_GAP_DAYS) -> tuple[pd.DataFrame, dict]:
    """Drop impossible bars and report gaps / suspicious rows. Returns (clean frame, report dict)."""
    rep: dict[str, object] = {"ticker": ticker, "rows_raw": int(len(df))}
    px = ["open", "high", "low", "close"]
    d = df.dropna(subset=px)
    rep["dropped_nan"] = int(len(df) - len(d))
    pos = (d[px] > 0).all(axis=1)
    rep["dropped_nonpositive"] = int((~pos).sum())
    d = d[pos]
    tol = 1e-9
    bad = (d["low"] > d[["open", "close"]].min(axis=1) * (1 + tol)) | \
          (d["high"] < d[["open", "close"]].max(axis=1) * (1 - tol))
    rep["dropped_ohlc_inconsistent"] = int(bad.sum())
    d = d[~bad].copy()
    rep["rows_clean"] = int(len(d))
    if len(d) >= 2:
        gaps = d.index.to_series().diff().dt.days.iloc[1:]
        rep["gaps_gt_%dd" % max_gap_days] = int((gaps > max_gap_days).sum())
        rep["max_gap_days"] = int(gaps.max())
        rep["max_gap_end"] = gaps.idxmax().date().isoformat()
        rep["moves_gt_10pct"] = int((d["close"].pct_change().abs() > 0.105).sum())
    else:
        rep["gaps_gt_%dd" % max_gap_days] = 0
        rep["max_gap_days"] = 0
        rep["max_gap_end"] = ""
        rep["moves_gt_10pct"] = 0
    rep["zero_volume_rows"] = int((d["volume"].fillna(0) <= 0).sum())
    rep["first_date"] = d.index[0].date().isoformat() if len(d) else ""
    rep["last_date"] = d.index[-1].date().isoformat() if len(d) else ""
    return d, rep


# ----------------------------------------------------------------------------
# Synthetic data (smoke test only)
# ----------------------------------------------------------------------------
def make_synthetic_ohlcv(n_bars: int, seed: int, start_price: float, daily_vol: float = 0.018,
                         drift: float = 0.0002, start: str = "2015-01-05", steps: int = 27) -> pd.DataFrame:
    """Geometric random walk with an intraday sub-path so that highs/lows are realistic.
    Used ONLY by --smoke-test. Has no relation to any real market."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_bars)
    gap = rng.normal(0.0, daily_vol * 0.4, n_bars)                         # overnight gap (log)
    intra = rng.standard_t(df=4, size=(n_bars, steps)) * (daily_vol * 0.9 / math.sqrt(steps)) / math.sqrt(2.0)
    intra += drift / steps
    path = np.cumsum(intra, axis=1)                                        # log path relative to the open
    day_ret = gap + path[:, -1]
    logc = math.log(start_price) + np.cumsum(day_ret)
    logo = logc - path[:, -1]
    logh = logo + np.maximum(0.0, path.max(axis=1))
    logl = logo + np.minimum(0.0, path.min(axis=1))
    vol = np.round(rng.lognormal(mean=math.log(2e6), sigma=0.6, size=n_bars))
    df = pd.DataFrame({"open": np.exp(logo), "high": np.exp(logh), "low": np.exp(logl),
                       "close": np.exp(logc), "volume": vol}, index=dates)
    df.index.name = "date"
    return df.round(2)


def make_smoke_universe(seed: int, n_bars: int = 1500) -> dict[str, pd.DataFrame]:
    specs = [("SYN_STOCK_A", 100.0, 0.020), ("SYN_STOCK_B", 45.0, 0.025),
             ("SYN_STOCK_C", 650.0, 0.018), ("^SYN_INDEX", 12000.0, 0.011)]
    return {name: make_synthetic_ohlcv(n_bars, seed + i, px, vol) for i, (name, px, vol) in enumerate(specs)}


# ----------------------------------------------------------------------------
# Market container
# ----------------------------------------------------------------------------
def compute_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    prev_c = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(period).mean().to_numpy()


class Market:
    """Numpy views of one instrument's cleaned OHLCV plus lazily computed helpers."""

    def __init__(self, ticker: str, df: pd.DataFrame):
        self.ticker = ticker
        self.kind: InstrumentKind = instrument_kind(ticker)
        self.df = df
        self.dates = df.index
        self.o = df["open"].to_numpy(dtype=float)
        self.h = df["high"].to_numpy(dtype=float)
        self.l = df["low"].to_numpy(dtype=float)
        self.c = df["close"].to_numpy(dtype=float)
        self.n = len(self.c)
        self.atr = compute_atr(self.h, self.l, self.c, 14)
        self.years = max((self.dates[-1] - self.dates[0]).days / 365.25, 1.0 / 252.0) if self.n > 1 else 1.0 / 252.0
        self._ph: dict[int, np.ndarray] = {}
        self._pl: dict[int, np.ndarray] = {}

    def prior_high(self, L: int) -> np.ndarray:
        """max(high[t-L .. t-1]) — excludes the current bar (no self-reference)."""
        if L not in self._ph:
            self._ph[L] = pd.Series(self.h).rolling(L).max().shift(1).to_numpy()
        return self._ph[L]

    def prior_low(self, L: int) -> np.ndarray:
        if L not in self._pl:
            self._pl[L] = pd.Series(self.l).rolling(L).min().shift(1).to_numpy()
        return self._pl[L]


# ----------------------------------------------------------------------------
# Trade records, fills and exits
# ----------------------------------------------------------------------------
@dataclass
class Trade:
    ticker: str
    rule: str
    side: str
    signal_idx: int
    entry_idx: int
    exit_idx: int
    entry_px: float
    exit_px: float
    stop: float
    target: float
    risk_frac: float
    exit_reason: str
    entry_kind: str


@dataclass
class Fill:
    status: str          # 'filled' | 'cancelled' | 'expired'
    idx: int
    px: float = float("nan")


def fill_limit_order(m: Market, start: int, end: int, level: float, side: str, stop: float) -> Fill:
    """Resting limit order at `level`, live on bars start..end (inclusive).
    Long: fills when low <= level (at min(open, level)); cancelled if a bar opens at/below the stop.
    Short: mirror."""
    last = min(end, m.n - 1)
    for k in range(start, last + 1):
        if side == "long":
            if m.o[k] <= stop:
                return Fill("cancelled", k)
            if m.l[k] <= level:
                return Fill("filled", k, min(m.o[k], level))
        else:
            if m.o[k] >= stop:
                return Fill("cancelled", k)
            if m.h[k] >= level:
                return Fill("filled", k, max(m.o[k], level))
    return Fill("expired", last)


def simulate_exit(m: Market, entry_idx: int, side: str, stop: float, target: float,
                  max_hold: int, entry_kind: str) -> tuple[int, float, str]:
    """Walk forward from the entry and return (exit_idx, exit_px, reason).  See module docstring
    for the conventions (stop-first on ambiguous bars, gaps filled at the open, time exit at close)."""
    n = m.n
    first = entry_idx + 1 if entry_kind == "close" else entry_idx
    last = entry_idx + max_hold - (0 if entry_kind == "close" else 1)
    has_stop = math.isfinite(stop)
    has_target = math.isfinite(target)
    for j in range(first, min(last, n - 1) + 1):
        oj, hj, lj = m.o[j], m.h[j], m.l[j]
        target_ok = has_target and not (entry_kind == "limit" and j == entry_idx)
        if side == "long":
            if target_ok and oj >= target:
                return j, oj, "target"
            if has_stop and oj <= stop:
                return j, oj, "stop"
            if has_stop and lj <= stop:
                return j, stop, "stop"
            if target_ok and hj >= target:
                return j, target, "target"
        else:
            if target_ok and oj <= target:
                return j, oj, "target"
            if has_stop and oj >= stop:
                return j, oj, "stop"
            if has_stop and hj >= stop:
                return j, stop, "stop"
            if target_ok and lj <= target:
                return j, target, "target"
    if last > n - 1:
        return n - 1, m.c[n - 1], "end_of_data"
    return last, m.c[last], "time"


class RuleStats:
    """Counts of signals that were skipped (and why) — reported for transparency."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str, str], int] = {}

    def bump(self, rule: str, side: str, reason: str) -> None:
        key = (rule, side, reason)
        self.counts[key] = self.counts.get(key, 0) + 1

    def frame(self) -> pd.DataFrame:
        rows = [{"rule": r, "side": s, "reason": why, "count": n} for (r, s, why), n in sorted(self.counts.items())]
        return pd.DataFrame(rows, columns=["rule", "side", "reason", "count"])


def make_trade(m: Market, cfg: Config, stats: RuleStats, rule: str, side: str, signal_idx: int,
               entry_idx: int, entry_px: float, entry_kind: str, stop: float, target: float,
               max_hold: int) -> Trade | None:
    """Validate risk, simulate the exit and build a Trade (or None if the setup is skipped)."""
    if math.isfinite(stop):
        if (side == "long" and entry_px <= stop) or (side == "short" and entry_px >= stop):
            stats.bump(rule, side, "entry_beyond_stop")
            return None
        risk_frac = abs(entry_px - stop) / entry_px
    else:
        atr = m.atr[signal_idx]
        if not math.isfinite(atr) or atr <= 0:
            stats.bump(rule, side, "no_atr")
            return None
        risk_frac = atr / entry_px
    if risk_frac < cfg.min_risk_pct / 100.0:
        stats.bump(rule, side, "risk_below_min")
        return None
    exit_idx, exit_px, reason = simulate_exit(m, entry_idx, side, stop, target, max_hold, entry_kind)
    if entry_kind == "close" and exit_idx <= entry_idx:
        stats.bump(rule, side, "no_bars_after_entry")
        return None
    return Trade(m.ticker, rule, side, signal_idx, entry_idx, exit_idx, float(entry_px), float(exit_px),
                 float(stop), float(target), float(risk_frac), reason, entry_kind)


def _target(side: str, entry_px: float, stop: float, k: float) -> float:
    risk = abs(entry_px - stop)
    return entry_px + k * risk if side == "long" else entry_px - k * risk


def _busy(cfg: Config, t: int, busy_until: int) -> bool:
    """True while a position / working order of this rule-side-instrument is still live.
    A signal on the bar the previous trade exits (t == busy_until) is allowed: the exit happens at that
    bar's close and any new entry happens at that close or later."""
    return (not cfg.allow_overlap) and t < busy_until


# ----------------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------------
def rule_r1_sweep(m: Market, cfg: Config, stats: RuleStats) -> list[Trade]:
    """R1 – liquidity sweep reversal.
    Long: low[t] < low[t-1] and close[t] > low[t-1] (optionally close > open).
    Entry close[t] (default) or open[t+1]; stop = low[t]*(1-buffer); target = k R; time exit --hold bars."""
    trades: list[Trade] = []
    buf = cfg.stop_buffer
    for side in cfg.sides:
        busy_until = -1
        for t in range(1, m.n - 1):
            if _busy(cfg, t, busy_until):
                continue
            if side == "long":
                sig = m.l[t] < m.l[t - 1] and m.c[t] > m.l[t - 1] and (not cfg.r1_strict or m.c[t] > m.o[t])
            else:
                sig = m.h[t] > m.h[t - 1] and m.c[t] < m.h[t - 1] and (not cfg.r1_strict or m.c[t] < m.o[t])
            if not sig:
                continue
            if cfg.r1_entry == "close":
                entry_idx, entry_px, kind = t, m.c[t], "close"
            else:
                entry_idx, entry_px, kind = t + 1, m.o[t + 1], "open"
            stop = m.l[t] * (1 - buf) if side == "long" else m.h[t] * (1 + buf)
            target = _target(side, entry_px, stop, cfg.target_r)
            tr = make_trade(m, cfg, stats, RULE_R1, side, t, entry_idx, entry_px, kind, stop, target, cfg.hold)
            if tr is not None:
                trades.append(tr)
                busy_until = tr.exit_idx
    return trades


def rule_r2_fvg(m: Market, cfg: Config, stats: RuleStats) -> list[Trade]:
    """R2 – fair value gap trade.
    Bullish FVG at t: high[t-2] < low[t]; zone = [high[t-2], low[t]]. Buy limit at the upper edge (low[t]),
    live for --fvg-wait bars; stop = lower edge*(1-buffer); target k R; time exit --level-hold bars."""
    trades: list[Trade] = []
    buf = cfg.stop_buffer
    for side in cfg.sides:
        busy_until = -1
        for t in range(2, m.n - 1):
            if _busy(cfg, t, busy_until):
                continue
            if side == "long":
                if not (m.h[t - 2] < m.l[t]):
                    continue
                lower, upper = m.h[t - 2], m.l[t]
                level, stop = upper, lower * (1 - buf)
            else:
                if not (m.l[t - 2] > m.h[t]):
                    continue
                lower, upper = m.h[t], m.l[t - 2]
                level, stop = lower, upper * (1 + buf)
            if (upper - lower) / m.c[t] < cfg.fvg_min_gap_pct / 100.0:
                stats.bump(RULE_R2, side, "gap_below_min")
                continue
            fill = fill_limit_order(m, t + 1, t + cfg.fvg_wait, level, side, stop)
            if fill.status == "filled":
                target = _target(side, fill.px, stop, cfg.target_r)
                tr = make_trade(m, cfg, stats, RULE_R2, side, t, fill.idx, fill.px, "limit", stop, target, cfg.level_hold)
                if tr is not None:
                    trades.append(tr)
                    busy_until = tr.exit_idx
                else:
                    busy_until = fill.idx
            elif fill.status == "cancelled":
                stats.bump(RULE_R2, side, "order_cancelled_gap_through")
                busy_until = fill.idx
            else:
                stats.bump(RULE_R2, side, "order_expired_unfilled")
                busy_until = fill.idx
    return trades


def _last_candle(m: Market, t: int, lookback: int, bearish: bool) -> int | None:
    for j in range(t - 1, max(t - 1 - lookback, -1), -1):
        if (m.c[j] < m.o[j]) if bearish else (m.c[j] > m.o[j]):
            return j
    return None


def rule_r3_order_block(m: Market, cfg: Config, stats: RuleStats) -> list[Trade]:
    """R3 – order-block retest.
    Bullish: bar t makes a new --lookback-bar high; the last bearish candle b within --ob-lookback bars before t
    is the OB (range low[b]..high[b]; requires close[t] > high[b]). Buy limit at high[b] live for --ob-wait bars;
    stop = low[b]*(1-buffer); target k R; time exit --level-hold bars.  Each OB candle is used once."""
    trades: list[Trade] = []
    L = cfg.lookback
    buf = cfg.stop_buffer
    ph, pl = m.prior_high(L), m.prior_low(L)
    for side in cfg.sides:
        busy_until = -1
        used: set[int] = set()
        for t in range(L, m.n - 1):
            if _busy(cfg, t, busy_until):
                continue
            if side == "long":
                if not (m.h[t] > ph[t]):
                    continue
                b = _last_candle(m, t, cfg.ob_lookback, bearish=True)
                if b is None or b in used:
                    continue
                ob_lo, ob_hi = m.l[b], m.h[b]
                if ob_hi <= ob_lo or not (m.c[t] > ob_hi):
                    stats.bump(RULE_R3, side, "ob_invalid")
                    continue
                level, stop = ob_hi, ob_lo * (1 - buf)
            else:
                if not (m.l[t] < pl[t]):
                    continue
                b = _last_candle(m, t, cfg.ob_lookback, bearish=False)
                if b is None or b in used:
                    continue
                ob_lo, ob_hi = m.l[b], m.h[b]
                if ob_hi <= ob_lo or not (m.c[t] < ob_lo):
                    stats.bump(RULE_R3, side, "ob_invalid")
                    continue
                level, stop = ob_lo, ob_hi * (1 + buf)
            used.add(b)
            fill = fill_limit_order(m, t + 1, t + cfg.ob_wait, level, side, stop)
            if fill.status == "filled":
                target = _target(side, fill.px, stop, cfg.target_r)
                tr = make_trade(m, cfg, stats, RULE_R3, side, t, fill.idx, fill.px, "limit", stop, target, cfg.level_hold)
                if tr is not None:
                    trades.append(tr)
                    busy_until = tr.exit_idx
                else:
                    busy_until = fill.idx
            elif fill.status == "cancelled":
                stats.bump(RULE_R3, side, "order_cancelled_gap_through")
                busy_until = fill.idx
            else:
                stats.bump(RULE_R3, side, "order_expired_unfilled")
                busy_until = fill.idx
    return trades


def rule_r4a_sr_bounce(m: Market, cfg: Config, stats: RuleStats) -> list[Trade]:
    """R4a – classic support/resistance bounce.
    Long: S = min(low[t-1-L .. t-2]) (the 20-bar low known at t-1); close[t-1] in [S, S*(1+tol)];
    bar t reverses up (close[t] > close[t-1] and close[t] > open[t]). Enter open[t+1];
    stop = S*(1-buffer); target k R; time exit --level-hold bars."""
    trades: list[Trade] = []
    L = cfg.lookback
    tol = cfg.sr_tol_pct / 100.0
    buf = cfg.stop_buffer
    ph, pl = m.prior_high(L), m.prior_low(L)
    for side in cfg.sides:
        busy_until = -1
        for t in range(L + 1, m.n - 1):
            if _busy(cfg, t, busy_until):
                continue
            if side == "long":
                S = pl[t - 1]
                if not math.isfinite(S):
                    continue
                near = S <= m.c[t - 1] <= S * (1 + tol)
                rev = m.c[t] > m.c[t - 1] and m.c[t] > m.o[t]
                if not (near and rev):
                    continue
                stop = S * (1 - buf)
            else:
                R = ph[t - 1]
                if not math.isfinite(R):
                    continue
                near = R * (1 - tol) <= m.c[t - 1] <= R
                rev = m.c[t] < m.c[t - 1] and m.c[t] < m.o[t]
                if not (near and rev):
                    continue
                stop = R * (1 + buf)
            entry_px = m.o[t + 1]
            target = _target(side, entry_px, stop, cfg.target_r)
            tr = make_trade(m, cfg, stats, RULE_R4A, side, t, t + 1, entry_px, "open", stop, target, cfg.level_hold)
            if tr is not None:
                trades.append(tr)
                busy_until = tr.exit_idx
    return trades


def rule_r4b_breakout(m: Market, cfg: Config, stats: RuleStats) -> list[Trade]:
    """R4b – classic breakout: close[t] > max(high[t-L .. t-1]) (mirror for shorts).
    Enter open[t+1], hold --hold bars, no stop/target. Risk unit for R multiples = ATR14[t]/entry."""
    trades: list[Trade] = []
    L = cfg.lookback
    ph, pl = m.prior_high(L), m.prior_low(L)
    for side in cfg.sides:
        busy_until = -1
        for t in range(L, m.n - 1):
            if _busy(cfg, t, busy_until):
                continue
            sig = (m.c[t] > ph[t]) if side == "long" else (m.c[t] < pl[t])
            if not sig:
                continue
            tr = make_trade(m, cfg, stats, RULE_R4B, side, t, t + 1, m.o[t + 1], "open",
                            float("nan"), float("nan"), cfg.hold)
            if tr is not None:
                trades.append(tr)
                busy_until = tr.exit_idx
    return trades


RULE_FUNCS = {
    RULE_R1: rule_r1_sweep,
    RULE_R2: rule_r2_fvg,
    RULE_R3: rule_r3_order_block,
    RULE_R4A: rule_r4a_sr_bounce,
    RULE_R4B: rule_r4b_breakout,
}


# ----------------------------------------------------------------------------
# Trade frame with returns and costs
# ----------------------------------------------------------------------------
def build_trade_frame(trades: list[Trade], markets: dict[str, Market], cost: CostModel) -> pd.DataFrame:
    rows: list[dict] = []
    for tr in trades:
        m = markets[tr.ticker]
        gross = (tr.exit_px / tr.entry_px - 1.0) if tr.side == "long" else (tr.entry_px - tr.exit_px) / tr.entry_px
        row = {
            "ticker": tr.ticker, "rule": tr.rule, "side": tr.side,
            "signal_date": m.dates[tr.signal_idx], "entry_date": m.dates[tr.entry_idx], "exit_date": m.dates[tr.exit_idx],
            "signal_idx": tr.signal_idx, "entry_idx": tr.entry_idx, "exit_idx": tr.exit_idx,
            "bars_held": tr.exit_idx - tr.entry_idx, "entry_kind": tr.entry_kind,
            "entry_px": tr.entry_px, "exit_px": tr.exit_px, "stop": tr.stop, "target": tr.target,
            "risk_frac": tr.risk_frac, "exit_reason": tr.exit_reason, "ret_gross": gross,
        }
        for mode in COST_MODES[1:]:
            row[RET_COL[mode]] = cost.net_return(tr.side, tr.entry_px, tr.exit_px, mode, m.kind)
        for mode in COST_MODES:
            row[R_COL[mode]] = row[RET_COL[mode]] / tr.risk_frac
        rows.append(row)
    cols = ["ticker", "rule", "side", "signal_date", "entry_date", "exit_date", "signal_idx", "entry_idx", "exit_idx",
            "bars_held", "entry_kind", "entry_px", "exit_px", "stop", "target", "risk_frac", "exit_reason"] + \
           [RET_COL[mo] for mo in COST_MODES] + [R_COL[mo] for mo in COST_MODES]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def equity_max_drawdown(r_mult: np.ndarray, order_key: np.ndarray, risk_f: float) -> tuple[float, float]:
    """Equal-risk sizing: each trade risks `risk_f` of equity, so P&L = risk_f * R.  Trades are compounded in
    exit-date order (overlaps are ignored).  Returns (max drawdown, final equity multiple)."""
    if len(r_mult) == 0:
        return float("nan"), float("nan")
    order = np.argsort(order_key, kind="stable")
    growth = 1.0 + risk_f * r_mult[order]
    eq = np.cumprod(np.clip(growth, 1e-9, None))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min()), float(eq[-1])


def compute_metrics(ret: np.ndarray, r_mult: np.ndarray, order_key: np.ndarray, bars_held: np.ndarray,
                    years: float, risk_f: float) -> dict:
    """Per-trade statistics.  Sharpe = mean/std of trade returns x sqrt(trades per year): assumes trade returns
    are i.i.d., non-overlapping and that capital is idle between trades (zero risk-free rate)."""
    n = int(len(ret))
    out: dict[str, float | int] = {"n_trades": n}
    if n == 0:
        for k in ("win_rate", "avg_ret_pct", "median_ret_pct", "expectancy_R", "total_R", "profit_factor",
                  "sharpe_ann", "max_dd_pct", "final_equity", "t_stat", "avg_bars_held", "trades_per_year"):
            out[k] = float("nan")
        return out
    gp = ret[ret > 0].sum()
    gl = -ret[ret < 0].sum()
    sd = float(ret.std(ddof=1)) if n > 1 else float("nan")
    tpy = n / years
    mdd, final_eq = equity_max_drawdown(r_mult, order_key, risk_f)
    out.update({
        "win_rate": float((ret > 0).mean()),
        "avg_ret_pct": float(ret.mean() * 100),
        "median_ret_pct": float(np.median(ret) * 100),
        "expectancy_R": float(np.nanmean(r_mult)),
        "total_R": float(np.nansum(r_mult)),
        "profit_factor": float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else float("nan")),
        "sharpe_ann": float(ret.mean() / sd * math.sqrt(tpy)) if sd and sd > 0 else float("nan"),
        "max_dd_pct": mdd * 100,
        "final_equity": final_eq,
        "t_stat": float(ret.mean() / (sd / math.sqrt(n))) if sd and sd > 0 else float("nan"),
        "avg_bars_held": float(bars_held.mean()),
        "trades_per_year": float(tpy),
    })
    return out


def bootstrap_random_entry(trades: pd.DataFrame, markets: dict[str, Market], n_boot: int,
                           rng: np.random.Generator) -> dict:
    """Null distribution of the mean GROSS return of the same number of random-entry trades, matched by
    instrument, side, entry timing and the rule's realised holding periods (resampled with replacement).
    p-value is one-sided: P(random mean >= observed mean)."""
    obs = float(trades["ret_gross"].mean())
    total = len(trades)
    acc = np.zeros(n_boot)
    for (ticker, side, kind), g in trades.groupby(["ticker", "side", "entry_kind"], sort=False):
        m = markets[ticker]
        holds = g["bars_held"].to_numpy(dtype=int)
        k = len(g)
        hold_s = rng.choice(holds, size=(n_boot, k), replace=True)
        max_start = m.n - 1 - hold_s
        start = np.floor(rng.random((n_boot, k)) * (max_start + 1)).astype(int)
        start = np.minimum(start, max_start)
        entry = m.c[start] if kind == "close" else m.o[start]
        exit_ = m.c[start + hold_s]
        r = exit_ / entry - 1.0
        if side == "short":
            r = -r
        acc += r.sum(axis=1)
    means = acc / total
    return {"p_boot": float((means >= obs).mean()), "rand_mean_pct": float(means.mean() * 100),
            "rand_sd_pct": float(means.std(ddof=1) * 100) if n_boot > 1 else float("nan")}


def metrics_table(trades: pd.DataFrame, markets: dict[str, Market], cfg: Config, rng: np.random.Generator,
                  sample_years: float) -> pd.DataFrame:
    """Metrics for every rule x side x (instrument | POOLED) x cost mode, with a bootstrap p-value per group."""
    rows: list[dict] = []
    if trades.empty:
        return pd.DataFrame(rows)
    tickers = list(markets.keys())
    for rule in RULE_ORDER:
        for side in cfg.sides:
            sub = trades[(trades["rule"] == rule) & (trades["side"] == side)]
            for ticker in tickers + ["POOLED"]:
                g = sub if ticker == "POOLED" else sub[sub["ticker"] == ticker]
                years = sample_years if ticker == "POOLED" else markets[ticker].years
                boot = {"p_boot": float("nan"), "rand_mean_pct": float("nan"), "rand_sd_pct": float("nan")}
                if len(g) >= 2:
                    boot = bootstrap_random_entry(g, markets, cfg.n_boot, rng)
                order_key = g["exit_date"].to_numpy()
                bars = g["bars_held"].to_numpy(dtype=float)
                for mode in COST_MODES:
                    met = compute_metrics(g[RET_COL[mode]].to_numpy(dtype=float), g[R_COL[mode]].to_numpy(dtype=float),
                                          order_key, bars, years, cfg.risk_per_trade)
                    row = {"rule": rule, "side": side, "ticker": ticker, "cost_mode": mode}
                    row.update(met)
                    row.update(boot)
                    if len(g):
                        reasons = g["exit_reason"].value_counts()
                        row["pct_stop"] = float(reasons.get("stop", 0) / len(g) * 100)
                        row["pct_target"] = float(reasons.get("target", 0) / len(g) * 100)
                        row["pct_time"] = float((reasons.get("time", 0) + reasons.get("end_of_data", 0)) / len(g) * 100)
                    rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Descriptive statistics: buy & hold, FVG fill rates, level magnet
# ----------------------------------------------------------------------------
def buy_and_hold_stats(m: Market, holds: Iterable[int]) -> dict:
    c = m.c
    daily = c[1:] / c[:-1] - 1.0
    peak = np.maximum.accumulate(c)
    sd = daily.std(ddof=1) if len(daily) > 1 else float("nan")
    out = {
        "ticker": m.ticker, "first_date": m.dates[0].date().isoformat(), "last_date": m.dates[-1].date().isoformat(),
        "bars": m.n, "years": m.years, "total_return_pct": (c[-1] / c[0] - 1) * 100,
        "cagr_pct": ((c[-1] / c[0]) ** (1 / m.years) - 1) * 100,
        "ann_vol_pct": sd * math.sqrt(252) * 100,
        "sharpe_daily_ann": daily.mean() / sd * math.sqrt(252) if sd and sd > 0 else float("nan"),
        "max_dd_pct": (c / peak - 1).min() * 100,
    }
    for N in sorted(set(holds)):
        if m.n > N:
            r = c[N:] / c[:-N] - 1.0
            out[f"uncond_mean_{N}bar_pct"] = r.mean() * 100
            out[f"uncond_win_{N}bar"] = (r > 0).mean()
    return out


def fvg_fill_stats(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, horizons: Iterable[int],
                   min_gap: float) -> dict:
    """For every bullish/bearish FVG: how often price re-enters the gap (touch) or trades through its far edge (fill)
    within H bars.  Only gaps with H complete bars after them are counted for horizon H."""
    n = len(c)
    out: dict[str, float] = {}
    for side in ("bull", "bear"):
        if side == "bull":
            idx = np.where(h[:-2] < l[2:])[0] + 2
            lower, upper = h[idx - 2], l[idx]
        else:
            idx = np.where(l[:-2] > h[2:])[0] + 2
            lower, upper = h[idx], l[idx - 2]
        size = (upper - lower) / c[idx] if len(idx) else np.array([])
        keep = size >= min_gap
        idx, lower, upper = idx[keep], lower[keep], upper[keep]
        out[f"{side}_n"] = int(len(idx))
        out[f"{side}_median_gap_pct"] = float(np.median(size[keep]) * 100) if keep.any() else float("nan")
        for H in horizons:
            valid = idx <= n - 1 - H
            ii = idx[valid]
            out[f"{side}_n_{H}"] = int(len(ii))
            if len(ii) == 0 or n <= H:
                out[f"{side}_touch_{H}"] = float("nan")
                out[f"{side}_fill_{H}"] = float("nan")
                continue
            if side == "bull":
                fwd = sliding_window_view(l, H)[ii + 1].min(axis=1)
                touched, filled = fwd <= upper[valid], fwd <= lower[valid]
            else:
                fwd = sliding_window_view(h, H)[ii + 1].max(axis=1)
                touched, filled = fwd >= lower[valid], fwd >= upper[valid]
            out[f"{side}_touch_{H}"] = float(touched.mean())
            out[f"{side}_fill_{H}"] = float(filled.mean())
    return out


def shuffle_bars(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Surrogate series: each bar's geometry relative to the previous close is kept, but the order of bars is
    shuffled.  Destroys serial dependence (any 'memory' of yesterday's levels) while preserving the marginal
    distribution of gaps, ranges and returns."""
    cp = c[:-1]
    ro, rh, rl, rc = o[1:] / cp, h[1:] / cp, l[1:] / cp, c[1:] / cp
    perm = rng.permutation(len(cp))
    c2 = np.concatenate([[c[0]], c[0] * np.exp(np.cumsum(np.log(rc[perm])))])
    cprev = c2[:-1]
    o2 = np.concatenate([[o[0]], cprev * ro[perm]])
    h2 = np.concatenate([[h[0]], cprev * rh[perm]])
    l2 = np.concatenate([[l[0]], cprev * rl[perm]])
    return o2, h2, l2, c2


def _touch_rates(h: np.ndarray, l: np.ndarray) -> dict[str, float]:
    pdh = h[1:] >= h[:-1]
    pdl = l[1:] <= l[:-1]
    return {"touch_pdh": float(pdh.mean()), "touch_pdl": float(pdl.mean()),
            "touch_either": float((pdh | pdl).mean()), "touch_both": float((pdh & pdl).mean())}


def brownian_magnet_null(o: np.ndarray, c: np.ndarray, rng: np.random.Generator,
                         n_days: int = 20000, steps: int = 54) -> dict[str, float]:
    """Driftless Brownian intraday path (54 five-minute steps, 09:00-13:30) with overnight gaps, volatilities matched
    to the instrument's realised overnight / intraday log-return standard deviations."""
    so = float(np.nanstd(np.log(o[1:] / c[:-1])))
    si = float(np.nanstd(np.log(c / o)))
    gaps = rng.normal(0.0, so, n_days)
    path = np.cumsum(rng.normal(0.0, si / math.sqrt(steps), (n_days, steps)), axis=1)
    logc = np.cumsum(gaps + path[:, -1])
    logo = logc - path[:, -1]
    h = np.exp(logo + np.maximum(0.0, path.max(axis=1)))
    l = np.exp(logo + np.minimum(0.0, path.min(axis=1)))
    return {f"brownian_{k}": v for k, v in _touch_rates(h, l).items()}


def level_magnet_stats(m: Market, n_shuffle: int, rng: np.random.Generator) -> dict:
    """How often does the next session touch the previous day's high / low?  Compared with (a) shuffled-bar
    surrogates (mean, sd, z-score) and (b) a Brownian random-walk null.  Purely descriptive."""
    real = _touch_rates(m.h, m.l)
    sur = {k: np.empty(n_shuffle) for k in real}
    for i in range(n_shuffle):
        _, h2, l2, _ = shuffle_bars(m.o, m.h, m.l, m.c, rng)
        for k, v in _touch_rates(h2, l2).items():
            sur[k][i] = v
    out: dict[str, float | str | int] = {"ticker": m.ticker, "days": m.n - 1}
    for k, v in real.items():
        mu, sd = float(sur[k].mean()), float(sur[k].std(ddof=1)) if n_shuffle > 1 else float("nan")
        out[k] = v
        out[f"{k}_shuffle_mean"] = mu
        out[f"{k}_shuffle_sd"] = sd
        out[f"{k}_z"] = (v - mu) / sd if sd and sd > 0 else float("nan")
    out.update(brownian_magnet_null(m.o, m.c, rng))
    return out


def fvg_table(markets: dict[str, Market], cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """FVG fill statistics per instrument + pooled, with the shuffled-bar null for the fill/touch rates."""
    rows: list[dict] = []
    min_gap = cfg.fvg_min_gap_pct / 100.0
    pooled_num: dict[str, float] = {}
    pooled_den: dict[str, float] = {}
    for m in markets.values():
        real = fvg_fill_stats(m.o, m.h, m.l, m.c, FVG_HORIZONS, min_gap)
        null_acc: dict[str, list[float]] = {}
        for _ in range(cfg.n_shuffle):
            o2, h2, l2, c2 = shuffle_bars(m.o, m.h, m.l, m.c, rng)
            s = fvg_fill_stats(o2, h2, l2, c2, FVG_HORIZONS, min_gap)
            for k, v in s.items():
                if ("touch" in k or "fill" in k) and isinstance(v, float) and math.isfinite(v):
                    null_acc.setdefault(k, []).append(v)
        row: dict = {"ticker": m.ticker}
        row.update(real)
        for k, vals in null_acc.items():
            row[f"{k}_null"] = float(np.mean(vals))
        rows.append(row)
        for side in ("bull", "bear"):
            for H in FVG_HORIZONS:
                cnt = real[f"{side}_n_{H}"]
                for stat in ("touch", "fill"):
                    key = f"{side}_{stat}_{H}"
                    if cnt and math.isfinite(real[key]):
                        pooled_num[key] = pooled_num.get(key, 0.0) + real[key] * cnt
                        pooled_den[key] = pooled_den.get(key, 0.0) + cnt
                    nk = f"{key}_null"
                    if nk in row and cnt:
                        pooled_num[nk] = pooled_num.get(nk, 0.0) + row[nk] * cnt
                        pooled_den[nk] = pooled_den.get(nk, 0.0) + cnt
    if rows:
        pooled: dict = {"ticker": "POOLED"}
        for side in ("bull", "bear"):
            pooled[f"{side}_n"] = int(sum(r[f"{side}_n"] for r in rows))
            for H in FVG_HORIZONS:
                pooled[f"{side}_n_{H}"] = int(sum(r[f"{side}_n_{H}"] for r in rows))
        for k, num in pooled_num.items():
            pooled[k] = num / pooled_den[k] if pooled_den[k] else float("nan")
        rows.append(pooled)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
@dataclass
class Results:
    cfg: Config
    trades: pd.DataFrame
    metrics: pd.DataFrame
    baselines: pd.DataFrame
    fvg: pd.DataFrame
    magnet: pd.DataFrame
    quality: pd.DataFrame
    skipped: pd.DataFrame
    cost_table: pd.DataFrame
    sample_years: float
    runtime_s: float


def run_backtest(cfg: Config, frames: dict[str, pd.DataFrame], quality_rows: list[dict]) -> Results:
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    cost = CostModel.from_config(cfg)
    markets: dict[str, Market] = {}
    for ticker, df in frames.items():
        if len(df) < cfg.lookback + cfg.level_hold + 5:
            LOG.warning("%s: only %d bars after cleaning – skipped", ticker, len(df))
            continue
        markets[ticker] = Market(ticker, df)
    if not markets:
        raise SystemExit("no instrument has enough data – nothing to do")

    stats = RuleStats()
    all_trades: list[Trade] = []
    for ticker, m in markets.items():
        for rule in RULE_ORDER:
            trs = RULE_FUNCS[rule](m, cfg, stats)
            LOG.info("%-12s %-14s %5d trades", ticker, rule, len(trs))
            all_trades.extend(trs)
    trades = build_trade_frame(all_trades, markets, cost)

    first = min(m.dates[0] for m in markets.values())
    last = max(m.dates[-1] for m in markets.values())
    sample_years = max((last - first).days / 365.25, 1.0 / 252.0)

    LOG.info("computing metrics and bootstrap p-values (n_boot=%d)", cfg.n_boot)
    metrics = metrics_table(trades, markets, cfg, rng, sample_years)
    LOG.info("computing baselines, FVG fill rates and level-magnet statistics (n_shuffle=%d)", cfg.n_shuffle)
    baselines = pd.DataFrame([buy_and_hold_stats(m, [cfg.hold, cfg.level_hold]) for m in markets.values()])
    fvg = fvg_table(markets, cfg, rng)
    magnet = pd.DataFrame([level_magnet_stats(m, cfg.n_shuffle, rng) for m in markets.values()])

    # indicative round-trip cost table at each instrument's median price
    cost_rows = []
    for m in markets.values():
        med = float(np.median(m.c))
        row = {"ticker": m.ticker, "kind": m.kind, "median_price": med}
        for mode in COST_MODES[1:]:
            row[f"round_trip_{mode}_pct"] = cost.round_trip_pct(mode, med, m.kind)
        cost_rows.append(row)
    cost_table = pd.DataFrame(cost_rows)

    return Results(cfg, trades, metrics, baselines, fvg, magnet, pd.DataFrame(quality_rows), stats.frame(),
                   cost_table, sample_years, time.time() - t0)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _fmt(v: object, nd: int = 2) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return ""
        if math.isinf(v):
            return "inf"
        return f"{v:.{nd}f}"
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return str(v)[:10]
    return str(v)


def md_table(df: pd.DataFrame, nd: int = 2) -> str:
    if df is None or df.empty:
        return "_(no rows)_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, nd) for v in row) + " |")
    return "\n".join(lines)


def pooled_view(metrics: pd.DataFrame, modes: Iterable[str]) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    cols = ["rule", "side", "cost_mode", "n_trades", "win_rate", "avg_ret_pct", "median_ret_pct", "expectancy_R",
            "profit_factor", "sharpe_ann", "max_dd_pct", "t_stat", "p_boot", "rand_mean_pct", "avg_bars_held",
            "pct_stop", "pct_target", "pct_time"]
    v = metrics[(metrics["ticker"] == "POOLED") & (metrics["cost_mode"].isin(list(modes)))][cols].copy()
    v["win_rate"] = v["win_rate"] * 100
    return v.rename(columns={"win_rate": "win_pct"})


def console_table(res: Results) -> str:
    """Compact pooled table: gross vs headline-net vs futures-net average return per rule & side."""
    met = res.metrics
    if met.empty:
        return "(no trades)"
    base = met[(met["ticker"] == "POOLED") & (met["cost_mode"] == "gross")].set_index(["rule", "side"])
    net = met[(met["ticker"] == "POOLED") & (met["cost_mode"] == res.cfg.headline_mode)].set_index(["rule", "side"])
    fut = met[(met["ticker"] == "POOLED") & (met["cost_mode"] == "futures")].set_index(["rule", "side"])
    rows = []
    for key in base.index:
        b, nrow, f = base.loc[key], net.loc[key], fut.loc[key]
        rows.append({
            "rule": key[0], "side": key[1], "n": int(b["n_trades"]), "win%": b["win_rate"] * 100,
            "avg_gross%": b["avg_ret_pct"], f"avg_net_{res.cfg.headline_mode}%": nrow["avg_ret_pct"],
            "avg_net_futures%": f["avg_ret_pct"], "expR_net": nrow["expectancy_R"], "PF_net": nrow["profit_factor"],
            "Sharpe_net": nrow["sharpe_ann"], "maxDD_net%": nrow["max_dd_pct"], "t_net": nrow["t_stat"],
            "p_boot": b["p_boot"],
        })
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        return df.to_string(index=False, float_format=lambda x: f"{x:.2f}", na_rep="")


def write_outputs(res: Results) -> Path:
    cfg = res.cfg
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades_all.csv", index=False, date_format="%Y-%m-%d")
    res.metrics.to_csv(out / "metrics_long.csv", index=False)
    res.baselines.to_csv(out / "baselines_buy_and_hold.csv", index=False)
    res.fvg.to_csv(out / "fvg_fill_rates.csv", index=False)
    res.magnet.to_csv(out / "level_magnet.csv", index=False)
    res.quality.to_csv(out / "data_quality.csv", index=False)
    res.skipped.to_csv(out / "skipped_signals.csv", index=False)
    res.cost_table.to_csv(out / "cost_table.csv", index=False)
    (out / "run_config.json").write_text(json.dumps(cfg.to_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8")
    md = build_summary_md(res)
    (out / "results_summary.md").write_text(md, encoding="utf-8")
    return out / "results_summary.md"


def build_summary_md(res: Results) -> str:
    cfg = res.cfg
    met = res.metrics
    L: list[str] = []
    title = "SMC / ICT / S-R mechanical daily-bar backtest – Taiwan"
    L.append(f"# {title}\n")
    if cfg.smoke_test:
        L.append("> **SYNTHETIC — NOT REAL RESULTS.** This file was produced by `--smoke-test` on in-memory random-walk data. "
                 "Nothing here describes any real market. Its only purpose is to show the code runs end-to-end.\n")
    L.append(f"Generated {datetime.now().isoformat(timespec='seconds')} · toolkit v{__version__} · "
             f"runtime {res.runtime_s:.1f}s · sample span {res.sample_years:.2f} years\n")
    L.append("## Parameters\n")
    p = cfg.to_jsonable()
    L.append("```json\n" + json.dumps(p, indent=2, ensure_ascii=False) + "\n```\n")
    L.append("## Conventions\n")
    L.append("* Signals use bars `<= t` only. R1 enters at the signal close by default (`--r1-entry close`, assumes an order in the "
             "closing auction; use `next_open` for the conservative variant); R4a/R4b enter at the next open; R2/R3 are resting limit "
             "orders filled when a later bar's range touches the level (open if gapped through; cancelled if the bar opens beyond the stop).\n"
             "* Exits: stop-first on bars that span both stop and target; gaps filled at the open; time exits at the close of the N-th bar held.\n"
             "* `expectancy_R`: mean return / risk per trade, risk = |entry − stop| / entry (R4b: ATR14 / entry). "
             f"Equity curve for max drawdown: each trade risks {cfg.risk_per_trade*100:.1f}% of equity (P&L = risk × R), compounded in exit-date order.\n"
             "* `sharpe_ann`: mean/std of trade returns × √(trades per year); assumes i.i.d., non-overlapping trades and idle capital between trades.\n"
             "* `p_boot`: one-sided bootstrap p-value — share of random-entry samples (same instrument/side/entry timing, holding periods "
             f"resampled from the rule's realised holds, {cfg.n_boot} samples) whose mean GROSS return ≥ the rule's mean. "
             "`rand_mean_pct` is the mean of that null distribution.\n"
             "* Cost modes: `gross`; `cash_overnight` (commission ×2 + 0.3% tax + borrow fee on shorts); `cash_daytrade` "
             "(hypothetical 0.15% tax — only legal for same-day round trips, a daily-bar strategy cannot actually obtain it); "
             "`futures` (0.002% tax + NT$ commission per contract per side). Slippage is applied in all net modes.\n")
    L.append("## Indicative round-trip cost at each instrument's median price (%, incl. slippage)\n")
    L.append(md_table(res.cost_table, 3) + "\n")

    L.append("## Pooled results by rule and side\n")
    for mode in COST_MODES:
        L.append(f"### Cost mode: `{mode}`\n")
        L.append(md_table(pooled_view(met, [mode]).drop(columns=["cost_mode"]), 2) + "\n")

    if not met.empty:
        L.append("## R4a (S/R bounce) vs R4b (breakout), pooled, headline net mode\n")
        cmp_ = pooled_view(met, [cfg.headline_mode])
        cmp_ = cmp_[cmp_["rule"].isin([RULE_R4A, RULE_R4B])]
        L.append(md_table(cmp_.drop(columns=["cost_mode"]), 2) + "\n")

        L.append(f"## Per-instrument results (cost mode `{cfg.headline_mode}`)\n")
        cols = ["rule", "side", "ticker", "n_trades", "win_rate", "avg_ret_pct", "expectancy_R", "profit_factor",
                "sharpe_ann", "max_dd_pct", "t_stat", "p_boot"]
        per = met[(met["ticker"] != "POOLED") & (met["cost_mode"] == cfg.headline_mode)][cols].copy()
        per["win_rate"] = per["win_rate"] * 100
        per = per.rename(columns={"win_rate": "win_pct"})
        for rule in RULE_ORDER:
            L.append(f"### {rule}\n")
            L.append(f"_{RULE_DESCRIPTIONS[rule]}_\n")
            L.append(md_table(per[per["rule"] == rule].drop(columns=["rule"]), 2) + "\n")

    L.append("## Baselines: buy & hold and unconditional N-bar returns\n")
    L.append(md_table(res.baselines, 2) + "\n")

    L.append("## R2 descriptive: FVG touch / fill rates (fraction of gaps) vs shuffled-bar null\n")
    L.append("`touch` = price re-enters the gap zone; `fill` = price trades through the far edge; `_null` = mean over "
             f"{cfg.n_shuffle} shuffled-bar surrogates (same bar geometry, random order).\n")
    fcols = ["ticker", "bull_n", "bull_median_gap_pct"] + \
            [f"bull_{s}_{H}" for H in FVG_HORIZONS for s in ("touch", "fill")] + \
            [f"bull_{s}_{H}_null" for H in FVG_HORIZONS for s in ("touch", "fill")]
    fcols_b = ["ticker", "bear_n", "bear_median_gap_pct"] + \
              [f"bear_{s}_{H}" for H in FVG_HORIZONS for s in ("touch", "fill")] + \
              [f"bear_{s}_{H}_null" for H in FVG_HORIZONS for s in ("touch", "fill")]
    L.append("### Bullish FVGs\n")
    L.append(md_table(res.fvg[[c for c in fcols if c in res.fvg.columns]], 3) + "\n")
    L.append("### Bearish FVGs\n")
    L.append(md_table(res.fvg[[c for c in fcols_b if c in res.fvg.columns]], 3) + "\n")

    L.append("## Level magnet: does the next session touch the previous day's high / low?\n")
    L.append("Real frequency vs shuffled-bar surrogate (mean, sd, z) and a driftless Brownian intraday null with matched "
             "overnight/intraday volatility. Descriptive only – a high touch rate is expected under a random walk as well.\n")
    mcols = ["ticker", "days", "touch_pdh", "touch_pdh_shuffle_mean", "touch_pdh_z", "brownian_touch_pdh",
             "touch_pdl", "touch_pdl_shuffle_mean", "touch_pdl_z", "brownian_touch_pdl",
             "touch_either", "touch_either_shuffle_mean", "touch_either_z", "brownian_touch_either",
             "touch_both", "touch_both_shuffle_mean", "brownian_touch_both"]
    L.append(md_table(res.magnet[[c for c in mcols if c in res.magnet.columns]], 3) + "\n")

    L.append("## Data quality\n")
    L.append(md_table(res.quality, 0) + "\n")
    L.append("## Skipped signals (why a signal did not become a trade)\n")
    L.append(md_table(res.skipped, 0) + "\n")

    L.append("## Limitations (read before drawing conclusions)\n")
    L.append("* SMC/ICT concepts are normally applied intraday and discretionarily; these are mechanical daily proxies and "
             "cannot confirm or refute the discretionary practice.\n"
             "* Daily OHLC bars do not reveal the intrabar path: stop/target ordering is assumed conservatively (stop first).\n"
             "* Short-side results for cash stocks ignore Taiwan margin-short constraints (券源, 平盤下不得放空 for non-eligible "
             "names, forced buy-backs before ex-dividend/AGM dates) – treat them as indicative; the futures cost mode is the "
             "realistic venue for shorts on the index.\n"
             "* Futures mode ignores margin, basis, roll and the fact that single-stock futures exist only for some names.\n"
             "* Transaction-cost defaults must be checked against the user's broker and current law.\n"
             "* Many rules × instruments × sides are tested: expect some small p-values by chance (multiple comparisons).\n")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="backtest_smc_taiwan.py",
        description="Mechanical daily-bar proxies of SMC/ICT and classic S/R rules on Taiwan data with Taiwan costs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Yahoo-style tickers (2330.TW, ^TWII ...)")
    g.add_argument("--source", choices=["yfinance", "finmind", "csv"], default="yfinance")
    g.add_argument("--start", default="2015-01-01")
    g.add_argument("--end", default=date.today().isoformat())
    g.add_argument("--data-dir", default="./data", help="cache directory for downloaded data")
    g.add_argument("--csv-dir", default="./data_csv", help="--source csv: directory with <ticker>.csv files")
    g.add_argument("--finmind-token", default=None, help="FinMind API token (optional; env FINMIND_TOKEN also read)")
    g.add_argument("--finmind-index-id", default="TAIEX", help="FinMind data_id used for the index ticker")
    g.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
    g.add_argument("--no-adjust", action="store_true", help="do NOT scale OHLC by adj_close/close (dividends)")
    g.add_argument("--out-dir", default=None, help="output directory (default ./output, ./output_smoke for smoke test)")

    r = p.add_argument_group("rules")
    r.add_argument("--sides", nargs="+", choices=["long", "short"], default=["long", "short"])
    r.add_argument("--hold", type=int, default=3, help="time exit (bars) for R1 and R4b")
    r.add_argument("--level-hold", type=int, default=10, help="time exit (bars) for limit/level rules R2, R3, R4a")
    r.add_argument("--target-r", type=float, default=2.0, help="target as a multiple of risk")
    r.add_argument("--stop-buffer-bps", type=float, default=10.0, help="stop placed this far beyond the level")
    r.add_argument("--min-risk-pct", type=float, default=0.2, help="skip setups whose stop distance is below this %%")
    r.add_argument("--lookback", type=int, default=20, help="bars for 20-bar highs/lows (R3, R4)")
    r.add_argument("--sr-tol-pct", type=float, default=0.5, help="R4a: close must be within this %% of the level")
    r.add_argument("--fvg-wait", type=int, default=20, help="R2: bars a limit order stays live")
    r.add_argument("--fvg-min-gap-pct", type=float, default=0.0, help="R2: ignore gaps smaller than this %% of price")
    r.add_argument("--ob-lookback", type=int, default=10, help="R3: bars to search back for the OB candle")
    r.add_argument("--ob-wait", type=int, default=20, help="R3: bars a limit order stays live")
    r.add_argument("--r1-entry", choices=["close", "next_open"], default="close")
    r.add_argument("--r1-strict", action="store_true", help="R1: also require close > open (long) / close < open (short)")
    r.add_argument("--allow-overlap", action="store_true", help="allow overlapping positions per rule/side/instrument")

    c = p.add_argument_group("costs (fractions in %% unless stated)")
    c.add_argument("--instrument", choices=["cash", "futures"], default="cash", help="headline net cost mode")
    c.add_argument("--comm-rate", type=float, default=0.1425, help="commission %% per side before discount")
    c.add_argument("--comm-discount", type=float, default=0.6, help="commission discount factor (0.6 = 6折)")
    c.add_argument("--tax-rate", type=float, default=0.3, help="securities transaction tax %% on sell")
    c.add_argument("--daytrade-tax-rate", type=float, default=0.15, help="same-day tax %% (hypothetical scenario)")
    c.add_argument("--short-borrow-bps", type=float, default=8.0, help="one-off borrow fee for cash shorts (bps)")
    c.add_argument("--fut-tax-rate", type=float, default=0.002, help="futures tax %% of notional per side")
    c.add_argument("--fut-comm", type=float, default=50.0, help="futures commission NT$ per contract per side")
    c.add_argument("--fut-multiplier", type=float, default=None, help="contract multiplier (default auto: 50 index / 2000 stock)")
    c.add_argument("--slippage-bps", type=float, default=5.0, help="slippage per side in bps")
    c.add_argument("--slippage-ticks", type=float, default=0.0, help="if > 0, slippage per side in ticks instead of bps")
    c.add_argument("--risk-per-trade", type=float, default=1.0, help="%% of equity risked per trade for the equity curve")

    s = p.add_argument_group("statistics / run mode")
    s.add_argument("--n-boot", type=int, default=1000, help="bootstrap samples for random-entry p-values")
    s.add_argument("--n-shuffle", type=int, default=200, help="shuffled-bar surrogates for FVG / magnet nulls")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--quick", action="store_true", help="fewer tickers (4), 200 bootstraps, 50 shuffles")
    s.add_argument("--smoke-test", action="store_true", help="run on in-memory synthetic random walks (NO real results)")
    s.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def config_from_args(a: argparse.Namespace) -> Config:
    tickers = list(a.tickers)
    n_boot, n_shuffle = a.n_boot, a.n_shuffle
    if a.quick:
        tickers = tickers[:4] if len(tickers) > 4 else tickers
        n_boot, n_shuffle = min(n_boot, 200), min(n_shuffle, 50)
    out_dir = Path(a.out_dir) if a.out_dir else Path("./output_smoke" if a.smoke_test else "./output")
    return Config(
        tickers=tickers, source=a.source, start=a.start, end=a.end, data_dir=Path(a.data_dir), out_dir=out_dir,
        csv_dir=Path(a.csv_dir), finmind_token=a.finmind_token or os.environ.get("FINMIND_TOKEN"),
        finmind_index_id=a.finmind_index_id, refresh=a.refresh, adjust=not a.no_adjust,
        sides=list(a.sides), hold=a.hold, level_hold=a.level_hold, target_r=a.target_r,
        stop_buffer_bps=a.stop_buffer_bps, min_risk_pct=a.min_risk_pct, lookback=a.lookback, sr_tol_pct=a.sr_tol_pct,
        fvg_wait=a.fvg_wait, fvg_min_gap_pct=a.fvg_min_gap_pct, ob_lookback=a.ob_lookback, ob_wait=a.ob_wait,
        r1_entry=a.r1_entry, r1_strict=a.r1_strict, allow_overlap=a.allow_overlap,
        instrument=a.instrument, comm_rate=a.comm_rate / 100.0, comm_discount=a.comm_discount,
        tax_rate=a.tax_rate / 100.0, daytrade_tax_rate=a.daytrade_tax_rate / 100.0, short_borrow_bps=a.short_borrow_bps,
        fut_tax_rate=a.fut_tax_rate / 100.0, fut_comm=a.fut_comm, fut_multiplier=a.fut_multiplier,
        slippage_bps=a.slippage_bps, slippage_ticks=a.slippage_ticks, risk_per_trade=a.risk_per_trade / 100.0,
        n_boot=n_boot, n_shuffle=n_shuffle, seed=a.seed, quick=a.quick, smoke_test=a.smoke_test,
    )


SMOKE_BANNER = "=" * 78 + "\n  SMOKE TEST — SYNTHETIC RANDOM-WALK DATA — NOT REAL RESULTS\n  (checks only that the code executes end-to-end; ignore every number)\n" + "=" * 78


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s", stream=sys.stderr)
    cfg = config_from_args(args)
    np.random.seed(cfg.seed)  # legacy global seed, in case any dependency uses it

    frames: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict] = []
    if cfg.smoke_test:
        print(SMOKE_BANNER)
        raw = make_smoke_universe(cfg.seed)
        cfg.tickers = list(raw.keys())
    else:
        raw = {}
        for ticker in cfg.tickers:
            try:
                raw[ticker] = load_instrument(cfg, ticker)
            except Exception as exc:  # network / format problems must not kill the whole run
                LOG.error("could not load %s: %s", ticker, exc)
        if not raw:
            LOG.error("no data loaded. The machine must reach the data source (yfinance/FinMind) or use --source csv.")
            return 2
    for ticker, df in raw.items():
        clean, rep = validate_ohlcv(df, ticker)
        quality_rows.append(rep)
        LOG.info("%s: %d rows (%s..%s), dropped nan=%d nonpos=%d ohlc=%d, gaps>%dd=%d, moves>10%%=%d",
                 ticker, rep["rows_clean"], rep["first_date"], rep["last_date"], rep["dropped_nan"],
                 rep["dropped_nonpositive"], rep["dropped_ohlc_inconsistent"], MAX_GAP_DAYS,
                 rep[f"gaps_gt_{MAX_GAP_DAYS}d"], rep["moves_gt_10pct"])
        frames[ticker] = clean

    res = run_backtest(cfg, frames, quality_rows)
    summary_path = write_outputs(res)

    if cfg.smoke_test:
        print(SMOKE_BANNER)
    slip_txt = f"{cfg.slippage_ticks:g} tick(s)/side" if cfg.slippage_ticks > 0 else f"{cfg.slippage_bps:g} bp/side"
    print(f"\nPooled results by rule/side (n={len(res.trades)} trades, {len(frames)} instruments, "
          f"headline net = {cfg.headline_mode}, slippage {slip_txt})")
    print(console_table(res))
    print(f"\nOutputs written to {cfg.out_dir.resolve()}  (summary: {summary_path.name})")
    if cfg.smoke_test:
        print(SMOKE_BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
