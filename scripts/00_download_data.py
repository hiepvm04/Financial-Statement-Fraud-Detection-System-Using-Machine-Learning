from __future__ import annotations

from pathlib import Path

from src.config import RAW_DATA_DIR, VNSTOCK_API_KEY
from src.data import (
    collect_financial_dataset,
    load_symbols_from_excel,
    set_vnstock_api_key,
)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    symbols_file = project_root / "all_symbols.xlsx"

    if not symbols_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {symbols_file}")

    set_vnstock_api_key(VNSTOCK_API_KEY)

    symbols = load_symbols_from_excel(
        excel_path=symbols_file,
        top_n=500,
    )

    print(f"Số mã không thuộc nhóm tài chính: {len(symbols)}")
    print("10 mã đầu:", symbols[:10])

    raw_df, failed_symbols = collect_financial_dataset(
        symbols=symbols,
        save_path=RAW_DATA_DIR / "raw_data.xlsx",
    )

    print("\n=== KẾT QUẢ THU THẬP DỮ LIỆU ===")
    print("Shape:", raw_df.shape)
    print("Số mã lỗi/bỏ qua:", len(failed_symbols))
    print("5 mã lỗi đầu:", failed_symbols[:5])


if __name__ == "__main__":
    main()