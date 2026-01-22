import baostock as bs
import pandas as pd
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

# ==========================================
# 配置项
# ==========================================
START_DATE = "2012-01-01"
END_DATE = "2026-01-10"
SAVE_DIR = "./data"
LOG_FILE = "download_progress.log"
ADJUST_FLAG = "2"  # 复权方式："2" 前复权，"1" 后复权，"3" 不复权
SLEEP_EVERY = 10
SLEEP_SECONDS = 0.5


def iter_month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    # 生成从开始到结束的每月月初日期
    cur = start.to_period("M").to_timestamp()
    end_m = end.to_period("M").to_timestamp()
    while cur <= end_m:
        yield cur
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()


def last_trading_day_in_month(calendar: pd.DatetimeIndex, month_start: pd.Timestamp) -> Optional[pd.Timestamp]:
    # 取当月最后一个交易日用于选股
    month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
    dates = calendar[(calendar >= month_start) & (calendar <= month_end)]
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def first_trading_day_in_month(calendar: pd.DatetimeIndex, month_start: pd.Timestamp) -> Optional[pd.Timestamp]:
    # 取当月第一个交易日用于买入
    month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
    dates = calendar[(calendar >= month_start) & (calendar <= month_end)]
    if dates.empty:
        return None
    return pd.Timestamp(dates.min()).normalize()


def last_trading_day_before_or_in_month(
    calendar: pd.DatetimeIndex, month_start: pd.Timestamp, end: pd.Timestamp
) -> Optional[pd.Timestamp]:
    # 取当月最后一个交易日卖出，受回测结束日期约束
    month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
    limit = min(month_end, end.normalize())
    dates = calendar[(calendar >= month_start) & (calendar <= limit)]
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


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


def to_ts_code_bs(code: str) -> str:
    # 将 baostock 代码 sh.600000 转换为 600000.SH
    code = code.strip()
    if "." in code:
        market, num = code.split(".")
        return f"{num}.{market.upper()}"
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


def download_all_a_shares() -> None:
    # 1. 初始化并登录
    lg = bs.login()
    if lg.error_code != "0":
        print(f"登录失败: {lg.error_msg}")
        return

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. 下载交易日历并计算策略日期
    cal_rs = bs.query_trade_dates(start_date=START_DATE, end_date=END_DATE)
    cal_list = []
    while (cal_rs.error_code == "0") & cal_rs.next():
        cal_list.append(cal_rs.get_row_data())
    cal_df = pd.DataFrame(cal_list, columns=["date", "is_trading"])
    cal_df["date"] = pd.to_datetime(cal_df["date"])
    cal_df["is_trading"] = pd.to_numeric(cal_df["is_trading"], errors="coerce").fillna(0).astype(int)
    cal_df.to_csv(save_dir / "calendar.csv", index=False)

    trading_calendar = pd.DatetimeIndex(cal_df[cal_df["is_trading"] == 1]["date"].sort_values())
    select_dates, buy_dates, sell_dates = collect_strategy_dates(
        trading_calendar, pd.to_datetime(START_DATE), pd.to_datetime(END_DATE)
    )
    price_dates = set(buy_dates + sell_dates)
    select_date_set = set(select_dates)

    # 3. 准备输出文件
    prices_path = save_dir / "prices.csv"
    pb_path = save_dir / "pb.csv"
    st_path = save_dir / "st.csv"
    ensure_csv_headers(prices_path, ["date", "ts_code", "open", "close"])
    ensure_csv_headers(pb_path, ["date", "ts_code", "pb"])
    ensure_csv_headers(st_path, ["date", "ts_code", "is_st"])

    # 4. 获取全市场股票代码列表（使用最近一个交易日，避免非交易日返回空）
    last_trade_date = trading_calendar.max().strftime("%Y-%m-%d")
    stock_rs = bs.query_all_stock(day=last_trade_date)
    stock_list = []
    while (stock_rs.error_code == "0") & stock_rs.next():
        stock_list.append(stock_rs.get_row_data()[0])

    print(f"使用交易日 {last_trade_date}，发现全市场共 {len(stock_list)} 只股票，开始下载日线数据...")

    # 5. 断点续传
    downloaded_stocks = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            downloaded_stocks = {line.strip() for line in f.readlines()}

    fields = "date,code,open,close,pbMRQ,isST"
    for i, code in enumerate(stock_list, start=1):
        if code in downloaded_stocks:
            continue

        print(f"[{i}/{len(stock_list)}] 正在处理: {code}")

        res = bs.query_history_k_data_plus(
            code,
            fields,
            start_date=START_DATE,
            end_date=END_DATE,
            frequency="d",
            adjustflag=ADJUST_FLAG,
        )

        data_rows = []
        while (res.error_code == "0") & res.next():
            data_rows.append(res.get_row_data())

        if not data_rows:
            continue

        df = pd.DataFrame(data_rows, columns=["date", "code", "open", "close", "pb", "is_st"])
        df["date"] = pd.to_datetime(df["date"])
        df["ts_code"] = df["code"].apply(to_ts_code_bs)
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["pb"] = pd.to_numeric(df["pb"], errors="coerce")
        df["is_st"] = pd.to_numeric(df["is_st"], errors="coerce").fillna(0).astype(int)

        # 价格：仅保留买入/卖出日期
        price_out = df[df["date"].isin(price_dates)][["date", "ts_code", "open", "close"]]
        if not price_out.empty:
            price_out.to_csv(prices_path, mode="a", index=False, header=False)

        # PB：仅保留选股日期
        pb_out = df[df["date"].isin(select_date_set)][["date", "ts_code", "pb"]].dropna()
        if not pb_out.empty:
            pb_out.to_csv(pb_path, mode="a", index=False, header=False)

        # ST：仅保留选股日期
        st_out = df[df["date"].isin(select_date_set)][["date", "ts_code", "is_st"]]
        if not st_out.empty:
            st_out.to_csv(st_path, mode="a", index=False, header=False)

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{code}\n")

        if i % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECONDS)

    bs.logout()
    print(">>> 下载完成：prices.csv / pb.csv / st.csv / calendar.csv")


if __name__ == "__main__":
    download_all_a_shares()
