import baostock as bs
import os
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Iterable, Optional

import pandas as pd

# ==========================================
# 配置项
# ==========================================
START_DATE = "2012-01-01"
END_DATE = "2026-01-10"
SAVE_DIR = "./data"
LOG_FILE = "download_progress.log"
FAILED_LOG = "download_failed.log"
EMPTY_LOG = "download_empty.log"
STATS_FILE = "download_stats.txt"
ADJUST_FLAG = "2"  # 复权方式："2" 前复权，"1" 后复权，"3" 不复权
SLEEP_EVERY = 10
SLEEP_SECONDS = 1.0
MAX_WORKERS = 1
MAX_RETRY = 3
RETRY_SLEEP = 1.0
RETRY_FAILED = True


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


def fetch_stock_basic() -> pd.DataFrame:
    # 获取全部股票基础信息（含退市），用于构建历史股票池。
    rs = bs.query_stock_basic()
    rows = []
    while (rs.error_code == "0") & rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    columns = rs.fields if hasattr(rs, "fields") else ["code", "code_name", "ipoDate", "outDate", "type", "status"]
    return pd.DataFrame(rows, columns=columns)


def read_log_set(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f.readlines() if line.strip()}


def append_log(path: str, code: str, lock: Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{code}\n")


def process_code(
    code: str,
    price_dates: set[pd.Timestamp],
    select_date_set: set[pd.Timestamp],
    prices_path: Path,
    pb_path: Path,
    st_path: Path,
    log_lock: Lock,
    csv_lock: Lock,
    stats: dict,
    stats_lock: Lock,
) -> None:
    fields = "date,code,open,close,pbMRQ,isST"
    for attempt in range(1, MAX_RETRY + 1):
        try:
            res = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=START_DATE,
                end_date=END_DATE,
                frequency="d",
                adjustflag=ADJUST_FLAG,
            )
            if res.error_code != "0":
                raise RuntimeError(res.error_msg)
            data_rows = []
            while res.next():
                data_rows.append(res.get_row_data())
        except Exception:
            if attempt < MAX_RETRY:
                time.sleep(RETRY_SLEEP)
                continue
            with stats_lock:
                stats["failed"] += 1
            append_log(FAILED_LOG, code, log_lock)
            return

        if not data_rows:
            with stats_lock:
                stats["empty"] += 1
            append_log(EMPTY_LOG, code, log_lock)
            return

        df = pd.DataFrame(data_rows, columns=["date", "code", "open", "close", "pb", "is_st"])
        df["date"] = pd.to_datetime(df["date"])
        df["ts_code"] = df["code"].apply(to_ts_code_bs)
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["pb"] = pd.to_numeric(df["pb"], errors="coerce")
        df["is_st"] = pd.to_numeric(df["is_st"], errors="coerce").fillna(0).astype(int)

        price_out = df[df["date"].isin(price_dates)][["date", "ts_code", "open", "close"]]
        pb_out = df[df["date"].isin(select_date_set)][["date", "ts_code", "pb"]].dropna()
        st_out = df[df["date"].isin(select_date_set)][["date", "ts_code", "is_st"]]

        with csv_lock:
            if not price_out.empty:
                price_out.to_csv(prices_path, mode="a", index=False, header=False)
            if not pb_out.empty:
                pb_out.to_csv(pb_path, mode="a", index=False, header=False)
            if not st_out.empty:
                st_out.to_csv(st_path, mode="a", index=False, header=False)

        with stats_lock:
            stats["downloaded"] += 1
            stats["price_rows"] += int(len(price_out))
            stats["pb_rows"] += int(len(pb_out))
            stats["st_rows"] += int(len(st_out))

        append_log(LOG_FILE, code, log_lock)
        return


def worker(
    worker_id: int,
    queue: Queue,
    price_dates: set[pd.Timestamp],
    select_date_set: set[pd.Timestamp],
    prices_path: Path,
    pb_path: Path,
    st_path: Path,
    log_lock: Lock,
    csv_lock: Lock,
    stats: dict,
    stats_lock: Lock,
) -> None:
    for _ in range(MAX_RETRY):
        lg = bs.login()
        if lg.error_code == "0":
            break
        print(f"线程 {worker_id} 登录失败: {lg.error_msg}")
        time.sleep(RETRY_SLEEP)
    else:
        return

    processed = 0
    while True:
        try:
            code = queue.get_nowait()
        except Empty:
            break

        process_code(
            code,
            price_dates,
            select_date_set,
            prices_path,
            pb_path,
            st_path,
            log_lock,
            csv_lock,
            stats,
            stats_lock,
        )
        queue.task_done()
        processed += 1

        if processed % SLEEP_EVERY == 0:
            time.sleep(SLEEP_SECONDS)

    bs.logout()


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
    listing_path = save_dir / "listing.csv"
    ensure_csv_headers(prices_path, ["date", "ts_code", "open", "close"])
    ensure_csv_headers(pb_path, ["date", "ts_code", "pb"])
    ensure_csv_headers(st_path, ["date", "ts_code", "is_st"])
    ensure_csv_headers(listing_path, ["ts_code", "list_date", "delist_date"])

    # 4. 获取全市场股票代码列表（包含退市股票，避免幸存者偏差）
    last_trade_date = trading_calendar.max().strftime("%Y-%m-%d")
    stock_basic = fetch_stock_basic()
    if stock_basic.empty:
        stock_rs = bs.query_all_stock(day=last_trade_date)
        stock_list = []
        while (stock_rs.error_code == "0") & stock_rs.next():
            stock_list.append(stock_rs.get_row_data()[0])
        listing_df = pd.DataFrame(columns=["ts_code", "list_date", "delist_date"])
        listing_count = 0
    else:
        if "type" in stock_basic.columns:
            stock_basic = stock_basic[stock_basic["type"] == "1"]
        listing_df = stock_basic.copy()
        if "code" in listing_df.columns:
            listing_df["ts_code"] = listing_df["code"].apply(to_ts_code_bs)
        elif "ts_code" not in listing_df.columns:
            listing_df["ts_code"] = ""
        if "ipoDate" in listing_df.columns:
            listing_df["list_date"] = pd.to_datetime(listing_df["ipoDate"], errors="coerce")
        else:
            listing_df["list_date"] = pd.NaT
        if "outDate" in listing_df.columns:
            listing_df["delist_date"] = pd.to_datetime(listing_df["outDate"], errors="coerce")
        else:
            listing_df["delist_date"] = pd.NaT
        listing_df = listing_df[["ts_code", "list_date", "delist_date"]].dropna(subset=["ts_code"])
        listing_df.to_csv(listing_path, index=False)
        stock_list = stock_basic["code"].dropna().tolist()
        listing_count = len(listing_df)

    bs.logout()

    print(
        f"使用交易日 {last_trade_date}，发现全市场共 {len(stock_list)} 只股票（含退市），"
        f"listing 记录 {listing_count} 条，开始下载日线数据..."
    )

    # 5. 断点续传
    downloaded_stocks = read_log_set(LOG_FILE)
    failed_stocks = read_log_set(FAILED_LOG)
    empty_stocks = read_log_set(EMPTY_LOG)

    if RETRY_FAILED:
        skip_set = downloaded_stocks.union(empty_stocks)
    else:
        skip_set = downloaded_stocks.union(failed_stocks).union(empty_stocks)

    todo_list = [code for code in stock_list if code not in skip_set]

    print(f"本次待下载 {len(todo_list)} 只股票，已完成 {len(downloaded_stocks)} 只")

    # 6. 多线程下载
    code_queue: Queue = Queue()
    for code in todo_list:
        code_queue.put(code)

    log_lock = Lock()
    csv_lock = Lock()
    stats_lock = Lock()
    stats = {
        "total": len(todo_list),
        "downloaded": 0,
        "empty": 0,
        "failed": 0,
        "price_rows": 0,
        "pb_rows": 0,
        "st_rows": 0,
        "listing_rows": listing_count,
    }

    threads: list[Thread] = []
    for i in range(MAX_WORKERS):
        t = Thread(
            target=worker,
            args=(
                i + 1,
                code_queue,
                price_dates,
                select_date_set,
                prices_path,
                pb_path,
                st_path,
                log_lock,
                csv_lock,
                stats,
                stats_lock,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # 7. 统计输出
    summary = (
        f"total={stats['total']}\n"
        f"downloaded={stats['downloaded']}\n"
        f"empty={stats['empty']}\n"
        f"failed={stats['failed']}\n"
        f"price_rows={stats['price_rows']}\n"
        f"pb_rows={stats['pb_rows']}\n"
        f"st_rows={stats['st_rows']}\n"
        f"listing_rows={stats['listing_rows']}\n"
    )
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write(summary)

    print("下载统计：")
    print(summary)
    print(">>> 下载完成：prices.csv / pb.csv / st.csv / calendar.csv / listing.csv")


if __name__ == "__main__":
    download_all_a_shares()
