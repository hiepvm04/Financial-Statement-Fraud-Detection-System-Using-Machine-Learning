from __future__ import annotations

import os
import time
from typing import Iterable

import numpy as np
import pandas as pd
from vnstock import Vnstock

from src.config import (
    RAW_DATA_DIR,
    VNSTOCK_API_KEY,
    DEFAULT_SOURCE,
    DEFAULT_START_YEAR,
    DEFAULT_END_YEAR,
    BATCH_SIZE,
    SLEEP_PER_SYMBOL,
    SLEEP_PER_BATCH,
)


def set_vnstock_api_key(api_key: str | None = None) -> None:
    """Set VNSTOCK API key vào environment."""
    key = api_key or VNSTOCK_API_KEY
    if key:
        os.environ["VNSTOCK_API_KEY"] = key


def load_symbols_from_excel(
    excel_path: str | os.PathLike,
    top_n: int = 500,
    symbol_col: str = "symbol",
    org_name_col: str = "organ_name",
) -> list[str]:
    """
    Đọc file all_symbols.xlsx và loại bỏ các doanh nghiệp tài chính.
    """
    df = pd.read_excel(excel_path)

    finance_keywords = ["ngân hàng", "chứng khoán", "bảo hiểm", "tài chính", "quỹ"]
    pattern = "|".join(finance_keywords)

    non_financial = df[~df[org_name_col].astype(str).str.lower().str.contains(pattern, na=False)]
    symbols = (
        non_financial[symbol_col]
        .dropna()
        .astype(str)
        .str.upper()
        .head(top_n)
        .tolist()
    )
    return symbols


def _normalize_keys(df: pd.DataFrame, fallback_symbol: str) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    cols = list(d.columns)
    low = {str(c).strip().lower(): c for c in cols}

    cp_col = next((low[k] for k in ["cp", "ticker", "symbol", "mã", "ma", "code"] if k in low), None)
    year_col = next(
        (
            low[k]
            for k in [
                "năm",
                "nam",
                "year",
                "fiscalyear",
                "reportyear",
                "yearreport",
                "year_report",
                "yearreporting",
                "yearreported",
            ]
            if k in low
        ),
        None,
    )

    if cp_col is None:
        d["CP"] = fallback_symbol
    else:
        d = d.rename(columns={cp_col: "CP"})

    if year_col is None:
        raise KeyError(f"Không tìm thấy cột Năm/year. Columns: {cols}")
    d = d.rename(columns={year_col: "Năm"})

    d["CP"] = d["CP"].astype(str).str.upper()
    d["Năm"] = pd.to_numeric(d["Năm"], errors="coerce").astype("Int64")
    return d


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Thiếu cột. Cần một trong {candidates}")
    return None


def _get_series(df: pd.DataFrame, candidates: list[str], default: float = np.nan) -> pd.Series:
    col = _pick_col(df, candidates, required=False)
    if col is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series) -> np.ndarray:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)


def process_stock(symbol: str, source: str = DEFAULT_SOURCE) -> pd.DataFrame | None:
    try:
        stock = Vnstock().stock(symbol=symbol, source=source)

        bs = stock.finance.balance_sheet(period="year", lang="vi", dropna=True).copy()
        is_ = stock.finance.income_statement(period="year", lang="vi", dropna=True).copy()
        cf = stock.finance.cash_flow(period="year", dropna=True).copy()
        is_quarter = stock.finance.income_statement(period="quarter", lang="vi", dropna=True).copy()

        bs = _normalize_keys(bs, symbol)
        is_ = _normalize_keys(is_, symbol)
        cf = _normalize_keys(cf, symbol)
        is_quarter = _normalize_keys(is_quarter, symbol)

        if is_quarter.empty or "Kỳ" not in is_quarter.columns or "Lợi nhuận thuần" not in is_quarter.columns:
            print(f"SKIP {symbol}: thiếu dữ liệu quý để tính Fraud")
            return None

        if "Lợi nhuận thuần" not in is_.columns:
            print(f"SKIP {symbol}: thiếu dữ liệu năm để tính Fraud")
            return None

        quarters = (
            is_quarter.pivot_table(
                index=["CP", "Năm"],
                columns="Kỳ",
                values="Lợi nhuận thuần",
                aggfunc="sum",
            )
            .reset_index()
            .rename(columns={1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
        )

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q not in quarters.columns:
                quarters[q] = np.nan

        quarters = quarters.dropna(subset=["Q1", "Q2", "Q3", "Q4"], how="any").copy()
        if quarters.empty:
            print(f"SKIP {symbol}: không có năm nào đủ 4 quý")
            return None

        quarters["Lợi nhuận thuần (trước kiểm toán)"] = quarters[["Q1", "Q2", "Q3", "Q4"]].sum(axis=1)

        year_df = is_[["CP", "Năm", "Lợi nhuận thuần"]].rename(
            columns={"Lợi nhuận thuần": "Lợi nhuận thuần (sau kiểm toán)"}
        )

        profit_df = quarters.merge(year_df, on=["CP", "Năm"], how="inner")
        if profit_df.empty:
            print(f"SKIP {symbol}: không match được dữ liệu quý và năm")
            return None

        profit_df["Chênh lệch"] = (
            profit_df["Lợi nhuận thuần (sau kiểm toán)"]
            - profit_df["Lợi nhuận thuần (trước kiểm toán)"]
        )

        profit_df["Phần trăm chênh lệch"] = np.where(
            profit_df["Lợi nhuận thuần (trước kiểm toán)"] == 0,
            np.nan,
            profit_df["Chênh lệch"] / profit_df["Lợi nhuận thuần (trước kiểm toán)"] * 100,
        )

        profit_df = profit_df.dropna(subset=["Phần trăm chênh lệch"]).copy()
        if profit_df.empty:
            print(f"SKIP {symbol}: không tính được phần trăm chênh lệch")
            return None

        profit_df["Fraud"] = (profit_df["Phần trăm chênh lệch"].abs() > 5).astype(int)

        df = (
            bs.merge(is_, on=["CP", "Năm"], how="outer", suffixes=("", "_is"))
            .merge(cf, on=["CP", "Năm"], how="outer", suffixes=("", "_cf"))
            .merge(profit_df[["CP", "Năm", "Fraud"]], on=["CP", "Năm"], how="inner")
            .sort_values(["CP", "Năm"])
            .reset_index(drop=True)
        )

        def lag(x: pd.Series) -> pd.Series:
            return x.groupby(df["CP"]).shift(1)

        TA = pd.to_numeric(df[_pick_col(df, ["TỔNG CỘNG TÀI SẢN (đồng)"])], errors="coerce")
        CA = pd.to_numeric(df[_pick_col(df, ["TÀI SẢN NGẮN HẠN (đồng)"])], errors="coerce")
        TL = pd.to_numeric(df[_pick_col(df, ["NỢ PHẢI TRẢ (đồng)"])], errors="coerce")
        EQ = pd.to_numeric(df[_pick_col(df, ["VỐN CHỦ SỞ HỮU (đồng)"])], errors="coerce")
        REV = pd.to_numeric(df[_pick_col(df, ["Doanh thu thuần"])], errors="coerce")
        COGS = pd.to_numeric(df[_pick_col(df, ["Giá vốn hàng bán"])], errors="coerce")

        CASH = _get_series(df, ["Tiền và tương đương tiền (đồng)"])
        AR = _get_series(df, ["Các khoản phải thu ngắn hạn (đồng)"])
        INV = _get_series(df, ["Hàng tồn kho ròng", "Hàng tồn kho, ròng (đồng)"])
        PPE = _get_series(df, ["Tài sản cố định (đồng)"])
        LTI = _get_series(df, ["Đầu tư dài hạn (đồng)"])
        CL = _get_series(df, ["Nợ ngắn hạn (đồng)"])

        sga1 = _pick_col(df, ["Chi phí bán hàng"], required=False)
        sga2 = _pick_col(df, ["Chi phí quản lý DN"], required=False)
        SGA = (
            (pd.to_numeric(df[sga1], errors="coerce") if sga1 else 0)
            + (pd.to_numeric(df[sga2], errors="coerce") if sga2 else 0)
        )

        ni_col = _pick_col(
            df,
            ["Lợi nhuận sau thuế của Cổ đông công ty mẹ (đồng)", "Lợi nhuận thuần"],
            required=False,
        )
        NI = pd.to_numeric(df[ni_col], errors="coerce") if ni_col else pd.Series(np.nan, index=df.index)

        CFO = _get_series(
            df,
            [
                "Lưu chuyển tiền tệ ròng từ các hoạt động SXKD",
                "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
                "Net cash inflows/outflows from operating activities",
                "Net cash inflows/outflows from operating activities_cf",
                "Net cash from operating activities",
                "Net cash from operating activities_cf",
            ],
        )

        DEP = _get_series(
            df,
            [
                "Khấu hao TSCĐ",
                "Khấu hao và phân bổ",
                "Depreciation and Amortisation",
                "Depreciation and Amortisation_cf",
                "Depreciation & Amortization",
                "Depreciation & Amortization_cf",
            ],
        )

        STDEBT = _get_series(df, ["Vay và nợ thuê tài chính ngắn hạn (đồng)"])
        LTDEBT = _get_series(df, ["Vay và nợ thuê tài chính dài hạn (đồng)"])

        df["Firm_Size"] = np.where(TA > 0, np.log(TA), np.nan)
        df["ROA"] = _safe_div(NI, TA)
        df["ROE"] = _safe_div(NI, EQ)
        df["Net_Profit_Margin"] = _safe_div(NI, REV)
        df["Gross_Profit_Margin"] = _safe_div(REV - COGS, REV)
        df["Sales_Growth"] = REV.groupby(df["CP"]).pct_change()
        df["Revenue_Growth"] = df["Sales_Growth"]
        df["Debt_to_Assets"] = _safe_div(TL, TA)
        df["Debt_to_Equity"] = _safe_div(TL, EQ)
        df["Receivables_to_Revenue"] = _safe_div(AR, REV)
        df["Receivables_to_Assets"] = _safe_div(AR, TA)
        df["Inventory_to_Assets"] = _safe_div(INV, TA)
        df["Current_Assets_to_Total_Assets"] = _safe_div(CA, TA)
        df["CFO_to_Assets"] = _safe_div(CFO, TA)
        df["CFO_to_Revenue"] = _safe_div(CFO, REV)
        df["Accruals_to_Assets"] = _safe_div(NI - CFO, TA)
        df["TATA"] = df["Accruals_to_Assets"]
        df["Working_Capital_to_Assets"] = _safe_div(CA - CL, TA)

        ar_rev = _safe_div(AR, REV)
        df["DSRI"] = _safe_div(pd.Series(ar_rev, index=df.index), lag(pd.Series(ar_rev, index=df.index)))

        gm = _safe_div(REV - COGS, REV)
        df["GMI"] = _safe_div(lag(pd.Series(gm, index=df.index)), pd.Series(gm, index=df.index))

        aq = 1 - _safe_div(CA + PPE, TA)
        df["AQI"] = _safe_div(pd.Series(aq, index=df.index), lag(pd.Series(aq, index=df.index)))

        df["SGI"] = _safe_div(REV, lag(pd.Series(REV, index=df.index)))

        dep_rate = _safe_div(DEP, PPE + DEP)
        df["DEPI"] = _safe_div(lag(pd.Series(dep_rate, index=df.index)), pd.Series(dep_rate, index=df.index))

        sga_rev = _safe_div(pd.Series(SGA, index=df.index), REV)
        df["SGAI"] = _safe_div(pd.Series(sga_rev, index=df.index), lag(pd.Series(sga_rev, index=df.index)))

        lev = _safe_div(TL, TA)
        df["LVGI"] = _safe_div(pd.Series(lev, index=df.index), lag(pd.Series(lev, index=df.index)))

        df["Delta_Receivables"] = _safe_div(AR - lag(pd.Series(AR, index=df.index)), TA)
        df["Delta_Inventory"] = _safe_div(INV - lag(pd.Series(INV, index=df.index)), TA)

        dREV = REV - lag(pd.Series(REV, index=df.index))
        dAR = AR - lag(pd.Series(AR, index=df.index))
        df["Delta_Cash_Sales"] = dREV - dAR

        df["Delta_ROA"] = df["ROA"] - lag(df["ROA"])
        df["Soft_Assets"] = _safe_div(TA - PPE - CASH, TA)

        WC = (CA - CASH) - (CL - STDEBT)
        NCO = (TA - CA - LTI) - (TL - CL - LTDEBT)
        FIN = LTI - (LTDEBT + STDEBT)

        dWC = WC - lag(pd.Series(WC, index=df.index))
        dNCO = NCO - lag(pd.Series(NCO, index=df.index))
        dFIN = FIN - lag(pd.Series(FIN, index=df.index))

        avgTA = (TA + lag(pd.Series(TA, index=df.index))) / 2
        df["RSST_Accruals"] = _safe_div(dWC + dNCO + dFIN, avgTA)

        cols_needed = [
            "CP", "Năm", "DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA",
            "RSST_Accruals", "Delta_Receivables", "Delta_Inventory", "Delta_Cash_Sales",
            "Delta_ROA", "Soft_Assets", "ROA", "ROE", "Net_Profit_Margin",
            "Gross_Profit_Margin", "Sales_Growth", "Revenue_Growth", "Debt_to_Assets",
            "Debt_to_Equity", "Receivables_to_Revenue", "Receivables_to_Assets",
            "Inventory_to_Assets", "Current_Assets_to_Total_Assets", "CFO_to_Assets",
            "CFO_to_Revenue", "Accruals_to_Assets", "Working_Capital_to_Assets",
            "Firm_Size", "Fraud"
        ]

        df = df[[c for c in cols_needed if c in df.columns]]
        df = df[
            (df["Năm"] >= DEFAULT_START_YEAR) & (df["Năm"] <= DEFAULT_END_YEAR)
        ].reset_index(drop=True)

        if df.empty or df["Fraud"].isna().all():
            print(f"SKIP {symbol}: không có Fraud hợp lệ")
            return None

        return df

    except Exception as e:
        print(f"Lỗi ở mã {symbol}: {e}")
        return None


def collect_financial_dataset(
    symbols: Iterable[str],
    batch_size: int = BATCH_SIZE,
    sleep_per_symbol: int = SLEEP_PER_SYMBOL,
    sleep_per_batch: int = SLEEP_PER_BATCH,
    save_path: str | os.PathLike | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Lặp qua nhiều mã cổ phiếu để tạo raw dataset.
    """
    all_dfs: list[pd.DataFrame] = []
    failed_symbols: list[str] = []

    symbols = list(symbols)

    for start in range(0, len(symbols), batch_size):
        batch = symbols[start:start + batch_size]
        print(f"\n=== Batch {start+1} - {start+len(batch)} ===")

        for i, sym in enumerate(batch, start + 1):
            print(f"[{i}/{len(symbols)}] Đang xử lý {sym}...")
            df_sym = process_stock(sym)

            if df_sym is not None and not df_sym.empty:
                all_dfs.append(df_sym)
                print(f"OK: {sym}")
            else:
                failed_symbols.append(sym)
                print(f"SKIP: {sym}")

            time.sleep(sleep_per_symbol)

        if start + batch_size < len(symbols):
            print(f"Nghỉ {sleep_per_batch} giây giữa các batch...")
            time.sleep(sleep_per_batch)

    all_data = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    if save_path is None:
        save_path = RAW_DATA_DIR / "raw_data.xlsx"

    all_data.to_excel(save_path, index=False)

    print("Số dòng:", len(all_data), "| Số cột:", all_data.shape[1] if not all_data.empty else 0)
    print("Số mã lỗi/bỏ qua:", len(failed_symbols))
    print("Danh sách mã lỗi:", failed_symbols)
    print(f"Đã lưu file: {save_path}")

    return all_data, failed_symbols