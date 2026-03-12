from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TARGET_COL
from src.preprocessing import (
    clip_outliers,
    fill_missing_values,
    load_raw_data,
    preprocess_dataset,
    summarize_data,
)


def main() -> None:
    raw_path = RAW_DATA_DIR / "raw_data.xlsx"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file raw data: {raw_path}\n"
            "Hãy chạy scripts/00_download_data.py trước."
        )

    df = load_raw_data(raw_path)

    print("=== RAW DATA SUMMARY ===")
    summary = summarize_data(df)
    print("Shape:", summary["shape"])
    print("Duplicate rows:", summary["duplicate_rows"])
    if "duplicate_cp_year" in summary:
        print("Duplicate [CP, Năm]:", summary["duplicate_cp_year"])

    print("\nTop missing values:")
    print(summary["missing"].head(10))

    df_filled = fill_missing_values(df, group_col="CP", exclude_cols=[TARGET_COL])
    print("\nMissing values sau fill:")
    print(df_filled.isna().sum().sort_values(ascending=False).head(10))

    df_clipped = clip_outliers(df_filled, lower_q=0.01, upper_q=0.99, exclude_cols=[TARGET_COL])
    print("\nShape sau clip outliers:", df_clipped.shape)

    processed_df = preprocess_dataset(
        df,
        save_path=PROCESSED_DATA_DIR / "processed_data.xlsx",
    )

    print("\n=== PROCESSED DATA SAVED ===")
    print("Path:", PROCESSED_DATA_DIR / "processed_data.xlsx")
    print("Shape:", processed_df.shape)

    df_check = pd.read_excel(PROCESSED_DATA_DIR / "processed_data.xlsx")
    print("Saved file shape:", df_check.shape)


if __name__ == "__main__":
    main()