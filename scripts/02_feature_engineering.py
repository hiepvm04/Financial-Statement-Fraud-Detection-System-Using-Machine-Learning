from __future__ import annotations

from src.config import PROCESSED_DATA_DIR, TOP_K_FEATURES
from src.features import (
    build_model_dataset,
    get_candidate_features,
    load_processed_data,
    rank_features_by_mutual_information,
    remove_multicollinearity,
    select_by_target_correlation,
    select_top_k_features,
    variance_filter,
)


def main() -> None:
    processed_path = PROCESSED_DATA_DIR / "processed_data.xlsx"

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file processed data: {processed_path}\n"
            "Hãy chạy scripts/01_preprocess_data.py trước."
        )

    df = load_processed_data(processed_path)

    print("=== PROCESSED DATA ===")
    print("Shape:", df.shape)

    X, y, feature_cols = get_candidate_features(df)
    print("Số candidate features:", len(feature_cols))

    X_vt, vt_features, removed_low_variance = variance_filter(X, threshold=0.0001)
    print("\nSau variance filter:", len(vt_features))
    print("Removed low variance:", removed_low_variance if removed_low_variance else "None")

    X_corr, corr_with_target, corr_selected_features = select_by_target_correlation(
        X_vt,
        y,
        threshold=0.02,
    )
    print("\nSố feature sau correlation filter:", len(corr_selected_features))

    X_final_candidates, removed_multicollinear = remove_multicollinearity(
        X_corr,
        corr_with_target,
        high_corr_threshold=0.80,
    )
    print("Removed multicollinearity:", removed_multicollinear if removed_multicollinear else "None")
    print("Số feature còn lại:", X_final_candidates.shape[1])

    mi_df = rank_features_by_mutual_information(X_final_candidates, y)
    selected_features = select_top_k_features(mi_df, top_k=TOP_K_FEATURES)

    print("\nTop features theo Mutual Information:")
    print(mi_df.head(TOP_K_FEATURES))
    print("\nSelected features:", selected_features)

    model_df, info = build_model_dataset(
        df,
        variance_threshold=0.0001,
        corr_threshold=0.02,
        high_corr_threshold=0.80,
        top_k=TOP_K_FEATURES,
        save_path=PROCESSED_DATA_DIR / "model_data.xlsx",
    )

    print("\n=== MODEL DATA SAVED ===")
    print("Path:", PROCESSED_DATA_DIR / "model_data.xlsx")
    print("Shape:", model_df.shape)
    print("Final selected features:", info["selected_features"])


if __name__ == "__main__":
    main()