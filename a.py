"""A股 PB 价值策略回测（按月换仓）。

CSV 输入（--data-dir）：
- prices.csv: date,ts_code,open,close
- pb.csv: date,ts_code,pb
- st.csv: date,ts_code,is_st   (is_st: 0/1，可为空)
- calendar.csv: date,is_trading (可选)
- listing.csv: ts_code,list_date,delist_date (可选)

日期格式：YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    start: str
    end: str
    top_pct: float = 0.10
    min_pb: float = 0.0
    pb_lag_months: int = 1
    pb_winsorize: bool = False
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    pb_zscore: bool = False


class DataSource:
    # 数据访问抽象层，用于将数据源与策略逻辑解耦。
    def get_trading_calendar(self, start: str, end: str) -> pd.DatetimeIndex:
        raise NotImplementedError

    def get_universe(self, date: str) -> pd.Index:
        raise NotImplementedError

    def get_prices(self, date: str, fields: Iterable[str]) -> pd.DataFrame:
        raise NotImplementedError

    def get_pb(self, date: str) -> pd.DataFrame:
        raise NotImplementedError

    def get_st(self, date: str) -> pd.Series:
        raise NotImplementedError


class CSVDataSource(DataSource):
    # 读取本地 CSV，方便在无实时 API 的情况下运行回测。
    def __init__(self, data_dir: str | Path) -> None:
        data_dir = Path(data_dir)
        self._prices = self._read_csv(data_dir / "prices.csv")
        self._pb = self._read_csv(data_dir / "pb.csv")
        self._st = self._read_csv(data_dir / "st.csv", optional=True)
        self._calendar = self._read_csv(data_dir / "calendar.csv", optional=True)
        self._listing = self._read_csv(data_dir / "listing.csv", optional=True)

        if not self._prices.empty:
            self._prices = self._prices.set_index(["date", "ts_code"]).sort_index()
        if not self._pb.empty:
            self._pb = self._pb.set_index(["date", "ts_code"]).sort_index()
        if not self._st.empty:
            self._st = self._st.set_index(["date", "ts_code"]).sort_index()

    def _read_csv(self, path: Path, optional: bool = False) -> pd.DataFrame:
        if optional and not path.exists():
            return pd.DataFrame()
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        if "list_date" in df.columns:
            df["list_date"] = pd.to_datetime(df["list_date"])
        if "delist_date" in df.columns:
            df["delist_date"] = pd.to_datetime(df["delist_date"])
        return df

    def get_trading_calendar(self, start: str, end: str) -> pd.DatetimeIndex:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        if not self._calendar.empty:
            cal = self._calendar
            cal = cal[cal["is_trading"] == 1]
            cal = cal[(cal["date"] >= start_ts) & (cal["date"] <= end_ts)]
            return pd.DatetimeIndex(cal["date"].sort_values())
        if self._prices.empty:
            return pd.DatetimeIndex([])
        dates = self._prices.index.get_level_values("date")
        dates = dates[(dates >= start_ts) & (dates <= end_ts)]
        return pd.DatetimeIndex(sorted(dates.unique()))

    def get_universe(self, date: str) -> pd.Index:
        date_ts = pd.to_datetime(date)
        if not self._listing.empty:
            df = self._listing
            listed = df[(df["list_date"] <= date_ts)]
            if "delist_date" in df.columns:
                listed = listed[(listed["delist_date"].isna()) | (listed["delist_date"] >= date_ts)]
            return pd.Index(listed["ts_code"].unique(), name="ts_code")
        if not self._prices.empty:
            return pd.Index(self._prices.index.get_level_values("ts_code").unique(), name="ts_code")
        if not self._pb.empty:
            return pd.Index(self._pb.index.get_level_values("ts_code").unique(), name="ts_code")
        return pd.Index([], name="ts_code")

    def get_prices(self, date: str, fields: Iterable[str]) -> pd.DataFrame:
        if self._prices.empty:
            return pd.DataFrame(columns=list(fields))
        date_ts = pd.to_datetime(date)
        try:
            df = self._prices.xs(date_ts, level="date")[list(fields)]
        except KeyError:
            return pd.DataFrame(columns=list(fields))
        df.index.name = "ts_code"
        return df

    def get_pb(self, date: str) -> pd.DataFrame:
        if self._pb.empty:
            return pd.DataFrame(columns=["pb"])
        date_ts = pd.to_datetime(date)
        try:
            df = self._pb.xs(date_ts, level="date")[["pb"]]
        except KeyError:
            return pd.DataFrame(columns=["pb"])
        df.index.name = "ts_code"
        return df

    def get_st(self, date: str) -> pd.Series:
        if self._st.empty:
            return pd.Series(dtype=bool)
        date_ts = pd.to_datetime(date)
        try:
            df = self._st.xs(date_ts, level="date")[["is_st"]]
        except KeyError:
            return pd.Series(dtype=bool)
        series = df["is_st"].astype(bool)
        series.index.name = "ts_code"
        return series


def winsorize_series(series: pd.Series, lower: float, upper: float) -> pd.Series:
    # 对因子做去极值，降低极端值影响。
    if series.empty:
        return series
    low = series.quantile(lower)
    high = series.quantile(upper)
    return series.clip(lower=low, upper=high)


def zscore_series(series: pd.Series) -> pd.Series:
    # 标准化为 z-score，避免量纲影响。
    if series.empty:
        return series
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


class PBValueBacktest:
    def __init__(self, data_source: DataSource, config: BacktestConfig) -> None:
        self.ds = data_source
        self.cfg = config

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        start = pd.to_datetime(self.cfg.start)
        end = pd.to_datetime(self.cfg.end)
        calendar = self.ds.get_trading_calendar(self.cfg.start, self.cfg.end)
        if calendar.empty:
            raise ValueError("Trading calendar is empty. Check data files.")

        # 汇总每期收益与持仓明细。
        results = []
        holdings = []

        for month_start in iter_month_starts(start, end):
            next_month_start = (month_start + pd.offsets.MonthBegin(1)).normalize()
            if next_month_start > end:
                break

            # 月末选股（PB 滞后），次月开盘买入，次月月末收盘卖出。
            select_date = last_trading_day_in_month(calendar, month_start)
            buy_date = first_trading_day_in_month(calendar, next_month_start)
            sell_date = last_trading_day_before_or_in_month(calendar, next_month_start, end)
            if select_date is None or buy_date is None or sell_date is None:
                continue

            lag_month_start = (month_start - pd.offsets.MonthBegin(self.cfg.pb_lag_months)).normalize()
            pb_date = last_trading_day_in_month(calendar, lag_month_start)
            if pb_date is None:
                continue

            picks = self.select_universe(select_date, pb_date)
            if picks.empty:
                continue

            buy_prices = self.ds.get_prices(buy_date, ["open"]).rename(columns={"open": "buy_open"})
            sell_prices = self.ds.get_prices(sell_date, ["close"]).rename(columns={"close": "sell_close"})
            px = picks.join(buy_prices, how="inner").join(sell_prices, how="inner")
            if px.empty:
                continue

            px["ret"] = px["sell_close"] / px["buy_open"] - 1.0
            period_ret = px["ret"].mean()

            results.append(
                {
                    "select_date": select_date,
                    "pb_date": pb_date,
                    "buy_date": buy_date,
                    "sell_date": sell_date,
                    "count": len(px),
                    "ret": period_ret,
                }
            )

            h_cols = ["pb"]
            if "factor" in px.columns:
                h_cols.append("factor")
            h = px[h_cols].copy()
            h["weight"] = 1.0 / len(px)
            h["select_date"] = select_date
            h["pb_date"] = pb_date
            h["buy_date"] = buy_date
            h["sell_date"] = sell_date
            h["ret"] = px["ret"]
            holdings.append(h.reset_index())

        res_df = pd.DataFrame(results)
        if res_df.empty:
            raise ValueError("No backtest results. Check data coverage.")
        res_df["equity"] = (1.0 + res_df["ret"]).cumprod()
        holdings_df = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
        return res_df, holdings_df

    def select_universe(self, select_date: pd.Timestamp, pb_date: pd.Timestamp) -> pd.DataFrame:
        # 过滤 PB 与 ST 股票，再选取 PB 最小的指定分位（PB 使用滞后月份）。
        pb = self.ds.get_pb(pb_date)
        if pb.empty:
            return pd.DataFrame(columns=["pb"])

        universe = self.ds.get_universe(select_date)
        pb = pb.loc[pb.index.intersection(universe)]
        pb = pb[pb["pb"] > self.cfg.min_pb]

        st = self.ds.get_st(select_date)
        if not st.empty:
            pb = pb[~pb.index.isin(st[st].index)]

        if pb.empty:
            return pd.DataFrame(columns=["pb"])

        factor = pb["pb"].copy()
        if self.cfg.pb_winsorize:
            factor = winsorize_series(factor, self.cfg.winsor_lower, self.cfg.winsor_upper)
        if self.cfg.pb_zscore:
            factor = zscore_series(factor)

        pb = pb.assign(factor=factor)

        n = max(1, int(len(pb) * self.cfg.top_pct))
        picks = pb.nsmallest(n, "factor")
        return picks


def iter_month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    # 生成从开始到结束的每月月初日期。
    cur = start.to_period("M").to_timestamp()
    end_m = end.to_period("M").to_timestamp()
    while cur <= end_m:
        yield cur
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()


def last_trading_day_in_month(calendar: pd.DatetimeIndex, month_start: pd.Timestamp) -> Optional[pd.Timestamp]:
    # 取当月最后一个交易日用于选股。
    month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
    dates = calendar[(calendar >= month_start) & (calendar <= month_end)]
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def first_trading_day_in_month(calendar: pd.DatetimeIndex, month_start: pd.Timestamp) -> Optional[pd.Timestamp]:
    # 取当月第一个交易日用于买入。
    month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
    dates = calendar[(calendar >= month_start) & (calendar <= month_end)]
    if dates.empty:
        return None
    return pd.Timestamp(dates.min()).normalize()


def last_trading_day_before_or_in_month(
    calendar: pd.DatetimeIndex, month_start: pd.Timestamp, end: pd.Timestamp
) -> Optional[pd.Timestamp]:
    # 取当月最后一个交易日卖出，受回测结束日期约束。
    month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
    limit = min(month_end, end.normalize())
    dates = calendar[(calendar >= month_start) & (calendar <= limit)]
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def performance_stats(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> dict:
    # 基于月度收益的统计指标，用于快速汇总。
    returns = returns.dropna()
    if returns.empty:
        return {}
    ann_ret = (1.0 + returns).prod() ** (12.0 / len(returns)) - 1.0
    ann_vol = returns.std(ddof=0) * np.sqrt(12.0)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    max_dd = drawdown.min()
    win_rate = (returns > 0).mean()

    downside = returns[returns < 0]
    if downside.empty:
        sortino = np.nan
    else:
        downside_vol = np.sqrt((downside ** 2).mean()) * np.sqrt(12.0)
        sortino = ann_ret / downside_vol if downside_vol != 0 else np.nan

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    beta = np.nan
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
        if len(aligned) >= 2:
            cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=0)[0, 1]
            var = np.var(aligned.iloc[:, 1], ddof=0)
            beta = cov / var if var != 0 else np.nan

    total_return = equity.iloc[-1] - 1.0

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "beta": beta,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_return": total_return,
    }


def max_drawdown_period(equity: pd.Series) -> tuple[int, int]:
    # 返回最大回撤对应的峰值索引与谷值索引。
    if equity.empty:
        return -1, -1
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    trough_idx = int(drawdown.idxmin())
    peak_value = equity.iloc[: trough_idx + 1].max()
    peak_candidates = equity.iloc[: trough_idx + 1]
    peak_idx = int(peak_candidates[peak_candidates == peak_value].index[-1])
    return peak_idx, trough_idx


def load_benchmark_csv(path: str) -> pd.DataFrame:
    # 读取基准指数 CSV，要求包含日期和收盘价列。
    df = pd.read_csv(path)
    columns = {c.lower(): c for c in df.columns}
    date_col = None
    for key in ("date", "trade_date"):
        if key in columns:
            date_col = columns[key]
            break
    if date_col is None and "date" in df.columns:
        date_col = "date"
    if date_col is None:
        raise ValueError("基准 CSV 缺少日期列（date/trade_date）")

    close_col = None
    for key in ("close", "adj_close", "adjclose", "price"):
        if key in columns:
            close_col = columns[key]
            break
    if close_col is None:
        # 兼容 Yahoo 风格列名
        for key in ("close", "adj close"):
            if key in columns:
                close_col = columns[key]
                break
    if close_col is None:
        raise ValueError("基准 CSV 缺少收盘价列（close/adj_close/price）")

    df = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date")
    return df


def prepare_benchmark_equity(benchmark: pd.DataFrame, dates: pd.Series) -> pd.Series:
    # 将基准指数对齐到回测日期，并归一化为权益曲线。
    if benchmark.empty:
        return pd.Series(dtype=float)
    bench = benchmark.set_index("date").sort_index()
    aligned = bench.reindex(pd.to_datetime(dates), method="ffill")
    aligned = aligned.dropna(subset=["close"])
    if aligned.empty:
        return pd.Series(dtype=float)
    base = aligned["close"].iloc[0]
    return aligned["close"] / base


def prepare_benchmark_returns(benchmark: pd.DataFrame, dates: pd.Series) -> pd.Series:
    # 将基准指数对齐到回测日期，并计算对应区间收益。
    if benchmark.empty:
        return pd.Series(dtype=float)
    bench = benchmark.set_index("date").sort_index()
    aligned = bench.reindex(pd.to_datetime(dates), method="ffill")
    aligned = aligned.dropna(subset=["close"])
    if aligned.empty:
        return pd.Series(dtype=float)
    returns = aligned["close"].pct_change()
    returns.index = aligned.index
    return returns.dropna()


def plot_equity_curve(
    results: pd.DataFrame,
    output_path: str,
    benchmark: Optional[pd.DataFrame] = None,
    benchmark_name: str = "纳斯达克100",
    show: bool = False,
) -> None:
    # 绘制权益曲线，并标注最大回撤区间，可选对比基准。
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        return

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    if results.empty:
        print("回测结果为空，无法绘图。")
        return

    plot_df = results.copy()
    if "sell_date" in plot_df.columns:
        plot_df = plot_df.sort_values("sell_date")
        x = pd.to_datetime(plot_df["sell_date"])
    else:
        plot_df = plot_df.sort_index()
        x = np.arange(len(plot_df))

    plot_df = plot_df.reset_index(drop=True)
    y = plot_df["equity"]

    fig, ax = plt.subplots(figsize=(11, 5))
    line, = ax.plot(x, y, label="策略", linewidth=1.6)

    if benchmark is not None and not benchmark.empty:
        bench_equity = prepare_benchmark_equity(benchmark, x)
        if not bench_equity.empty:
            ax.plot(x[: len(bench_equity)], bench_equity.values, label=benchmark_name, linewidth=1.4)

    peak_idx, trough_idx = max_drawdown_period(y.reset_index(drop=True))
    if peak_idx >= 0 and trough_idx >= 0:
        peak_x = x.iloc[peak_idx] if hasattr(x, "iloc") else x[peak_idx]
        trough_x = x.iloc[trough_idx] if hasattr(x, "iloc") else x[trough_idx]
        peak_y = y.iloc[peak_idx]
        trough_y = y.iloc[trough_idx]
        ax.axvline(peak_x, color="#d55e00", linestyle="--", linewidth=1)
        ax.axvline(trough_x, color="#d55e00", linestyle="--", linewidth=1)
        ax.fill_betweenx([trough_y, peak_y], peak_x, trough_x, color="#d55e00", alpha=0.12)
        ax.annotate(
            "最大回撤",
            xy=(trough_x, trough_y),
            xytext=(trough_x, peak_y * 0.95),
            arrowprops=dict(arrowstyle="->", color="#d55e00"),
            fontsize=9,
            color="#d55e00",
        )

    use_datetime = hasattr(x, "dtype") and np.issubdtype(x.dtype, np.datetime64)
    if use_datetime:
        import matplotlib.dates as mdates

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.format_xdata = mdates.DateFormatter("%Y-%m").__call__
        x_dt = pd.to_datetime(x)
        x_num = mdates.date2num(x_dt.dt.to_pydatetime())
    else:
        x_num = np.asarray(x, dtype=float)

    hover_dot = ax.scatter([], [], s=18, color="#d55e00", zorder=5, visible=False)
    hover_text = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color="#d55e00",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d55e00", alpha=0.8),
        visible=False,
    )

    x_data = line.get_xdata()
    y_data = line.get_ydata()

    def _on_move(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            if hover_text.get_visible():
                hover_text.set_visible(False)
                hover_dot.set_visible(False)
                fig.canvas.draw_idle()
            return
        idx = int(np.searchsorted(x_num, event.xdata))
        idx = max(0, min(idx, len(x_data) - 1))
        x_val = x_data[idx]
        y_val = y_data[idx]
        if use_datetime:
            x_display = pd.to_datetime(x_val).strftime("%Y-%m")
        else:
            x_display = f"{x_val:.0f}"
        hover_dot.set_offsets([[x_val, y_val]])
        hover_dot.set_visible(True)
        hover_text.xy = (x_val, y_val)
        ret_val = plot_df.loc[idx, "ret"] if "ret" in plot_df.columns else np.nan
        if pd.notna(ret_val):
            hover_text.set_text(f"{x_display}\n权益: {y_val:.2f}\n当月收益: {ret_val:.2%}")
        else:
            hover_text.set_text(f"{x_display}\n权益: {y_val:.2f}")
        hover_text.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_move)

    ax.set_title("PB 价值策略权益曲线")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def to_ts_code(code: str) -> str:
    code = code.strip()
    if code.endswith((".SH", ".SZ", ".BJ")):
        return code
    if code.startswith("6"):
        suffix = "SH"
    elif code.startswith(("8", "4")):
        suffix = "BJ"
    else:
        suffix = "SZ"
    return f"{code}.{suffix}"


def ensure_csv_headers(path: Path, columns: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def collect_strategy_dates(
    calendar: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], list[pd.Timestamp]]:
    select_dates: list[pd.Timestamp] = []
    buy_dates: list[pd.Timestamp] = []
    sell_dates: list[pd.Timestamp] = []
    for month_start in iter_month_starts(start, end):
        next_month_start = (month_start + pd.offsets.MonthBegin(1)).normalize()
        if next_month_start > end:
            break
        select_date = last_trading_day_in_month(calendar, month_start)
        buy_date = first_trading_day_in_month(calendar, next_month_start)
        sell_date = last_trading_day_before_or_in_month(calendar, next_month_start, end)
        if select_date is None or buy_date is None or sell_date is None:
            continue
        select_dates.append(select_date)
        buy_dates.append(buy_date)
        sell_dates.append(sell_date)
    return select_dates, buy_dates, sell_dates


def download_akshare_dataset(
    data_dir: str | Path,
    start: str,
    end: str,
    adjust: str,
    refresh: bool = False,
) -> None:
    # 用 AkShare 拉取数据并落地到 CSV，便于复现回测。
    # 免费接口通常不提供历史 ST 标记，这里生成空 st.csv（回测将跳过 ST 过滤）。
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("未安装 akshare，请先安装后再运行。") from exc

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    cal_path = data_dir / "calendar.csv"
    if refresh or not cal_path.exists():
        cal = ak.tool_trade_date_hist_sina()
        date_col = "trade_date" if "trade_date" in cal.columns else "date"
        cal["date"] = pd.to_datetime(cal[date_col])
        cal = cal[(cal["date"] >= pd.to_datetime(start)) & (cal["date"] <= pd.to_datetime(end))]
        cal = cal.sort_values("date")
        cal_out = cal[["date"]].copy()
        cal_out["is_trading"] = 1
        cal_out.to_csv(cal_path, index=False)

    calendar = pd.read_csv(cal_path)
    calendar["date"] = pd.to_datetime(calendar["date"])
    trading_calendar = pd.DatetimeIndex(calendar[calendar["is_trading"] == 1]["date"].sort_values())

    select_dates, buy_dates, sell_dates = collect_strategy_dates(
        trading_calendar, pd.to_datetime(start), pd.to_datetime(end)
    )
    price_dates = set(buy_dates + sell_dates)

    prices_path = data_dir / "prices.csv"
    pb_path = data_dir / "pb.csv"
    st_path = data_dir / "st.csv"

    if refresh:
        for path in (prices_path, pb_path, st_path):
            if path.exists():
                path.unlink()

    ensure_csv_headers(prices_path, ["date", "ts_code", "open", "close"])
    ensure_csv_headers(pb_path, ["date", "ts_code", "pb"])
    ensure_csv_headers(st_path, ["date", "ts_code", "is_st"])

    stock_info = ak.stock_info_a_code_name()
    codes = stock_info["code"].astype(str).tolist()

    start_str = pd.to_datetime(start).strftime("%Y%m%d")
    end_str = pd.to_datetime(end).strftime("%Y%m%d")

    for idx, code in enumerate(codes, start=1):
        ts_code = to_ts_code(code)

        try:
            hist = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust=adjust,
            )
        except Exception as exc:
            print(f"行情下载失败: {ts_code} {exc}")
            continue

        if not hist.empty:
            hist = hist.rename(columns={"日期": "date", "开盘": "open", "收盘": "close"})
            hist["date"] = pd.to_datetime(hist["date"])
            hist = hist[hist["date"].isin(price_dates)]
            if not hist.empty:
                out = hist[["date", "open", "close"]].copy()
                out["ts_code"] = ts_code
                out = out[["date", "ts_code", "open", "close"]]
                out.to_csv(prices_path, mode="a", index=False, header=False)

        try:
            pb_df = ak.stock_a_lg_indicator(symbol=code)
        except Exception as exc:
            print(f"PB 下载失败: {ts_code} {exc}")
            continue

        if not pb_df.empty and "pb" in pb_df.columns:
            date_col = "trade_date" if "trade_date" in pb_df.columns else "date"
            pb_df = pb_df.rename(columns={date_col: "date"})
            pb_df["date"] = pd.to_datetime(pb_df["date"])
            pb_df = pb_df[pb_df["date"].isin(select_dates)]
            if not pb_df.empty:
                out = pb_df[["date", "pb"]].copy()
                out["ts_code"] = ts_code
                out = out[["date", "ts_code", "pb"]]
                out.to_csv(pb_path, mode="a", index=False, header=False)

        if idx % 200 == 0:
            print(f"已处理 {idx} / {len(codes)} 只股票")


def required_files_exist(data_dir: str | Path) -> bool:
    data_dir = Path(data_dir)
    return (data_dir / "prices.csv").exists() and (data_dir / "pb.csv").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="A股月度 PB 价值策略回测")
    parser.add_argument("--data-dir", default="data", help="CSV 数据目录")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--out", default="backtest_result.csv")
    parser.add_argument("--save-holdings", default="holdings.csv")
    parser.add_argument("--pb-lag-months", type=int, default=1, help="PB 滞后月数（避免未来数据）")
    parser.add_argument("--plot", action="store_true", help="绘制权益曲线（默认开启）")
    parser.add_argument("--show-plot", action="store_true", help="绘图后弹窗显示（默认开启）")
    parser.add_argument("--no-plot", action="store_true", help="关闭绘图")
    parser.add_argument("--plot-out", default="equity_curve.png", help="权益曲线图片输出路径")
    parser.add_argument("--benchmark-csv", default="", help="基准指数 CSV（日期+收盘价）")
    parser.add_argument("--benchmark-name", default="纳斯达克100", help="基准名称")
    parser.add_argument("--winsorize-pb", action="store_true", help="对 PB 做去极值处理")
    parser.add_argument("--winsor-lower", type=float, default=0.01, help="去极值下分位")
    parser.add_argument("--winsor-upper", type=float, default=0.99, help="去极值上分位")
    parser.add_argument("--zscore-pb", action="store_true", help="对 PB 做标准化")
    parser.add_argument("--report-out", default="report.json", help="回测报告输出路径")
    parser.add_argument("--use-akshare", action="store_true", help="使用 AkShare 自动下载并生成 CSV")
    parser.add_argument("--download-akshare", action="store_true", help="强制重新下载 AkShare 数据")
    parser.add_argument("--adjust", default="qfq", help="复权方式：qfq/hfq/''")
    args = parser.parse_args()

    if args.use_akshare and (args.download_akshare or not required_files_exist(args.data_dir)):
        print("开始使用 AkShare 下载数据，耗时较长，请耐心等待。")
        download_akshare_dataset(
            data_dir=args.data_dir,
            start=args.start,
            end=args.end,
            adjust=args.adjust,
            refresh=args.download_akshare,
        )

    cfg = BacktestConfig(
        start=args.start,
        end=args.end,
        top_pct=0.10,
        min_pb=0.0,
        pb_lag_months=args.pb_lag_months,
        pb_winsorize=args.winsorize_pb,
        winsor_lower=args.winsor_lower,
        winsor_upper=args.winsor_upper,
        pb_zscore=args.zscore_pb,
    )
    ds = CSVDataSource(args.data_dir)
    bt = PBValueBacktest(ds, cfg)

    res_df, holdings_df = bt.run()
    benchmark = None
    benchmark_returns = None
    if args.benchmark_csv:
        benchmark = load_benchmark_csv(args.benchmark_csv)
        benchmark_returns = prepare_benchmark_returns(benchmark, res_df["sell_date"])

    stats = performance_stats(res_df["ret"], benchmark_returns)

    res_df.to_csv(args.out, index=False)
    if not holdings_df.empty:
        holdings_df.to_csv(args.save_holdings, index=False)

    print("回测期数:", len(res_df))
    print(f"年化收益率: {stats.get('ann_return', float('nan')):.4f}")
    print(f"年化波动率: {stats.get('ann_vol', float('nan')):.4f}")
    print(f"夏普比率: {stats.get('sharpe', float('nan')):.4f}")
    print(f"索提诺比率: {stats.get('sortino', float('nan')):.4f}")
    print(f"卡玛比率: {stats.get('calmar', float('nan')):.4f}")
    print(f"最大回撤: {stats.get('max_drawdown', float('nan')):.4f}")
    print(f"胜率: {stats.get('win_rate', float('nan')):.4f}")
    print(f"累计收益: {stats.get('total_return', float('nan')):.4f}")
    if args.benchmark_csv:
        print(f"Beta({args.benchmark_name}): {stats.get('beta', float('nan')):.4f}")

    report = {
        "periods": len(res_df),
        "start": args.start,
        "end": args.end,
        "pb_lag_months": cfg.pb_lag_months,
        "winsorize_pb": cfg.pb_winsorize,
        "winsor_lower": cfg.winsor_lower,
        "winsor_upper": cfg.winsor_upper,
        "zscore_pb": cfg.pb_zscore,
        "benchmark_name": args.benchmark_name if args.benchmark_csv else None,
        "metrics": stats,
    }
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.no_plot:
        plot_equity_curve(
            res_df,
            args.plot_out,
            benchmark,
            args.benchmark_name,
            show=True if not args.plot and not args.show_plot else args.show_plot,
        )


if __name__ == "__main__":
    main()
