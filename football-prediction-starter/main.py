from src.data_io import load_matches, time_split
from src.features import build_feature_table
from src.train import fit_model
from src.evaluate import evaluate_model
from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def main() -> None:
    data_path = PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else RAW_DATA_PATH
    matches = load_matches(data_path)
    feature_df = build_feature_table(matches)

    train_df, test_df = time_split(feature_df, test_ratio=0.2)
    model, feature_cols = fit_model(train_df)
    metrics = evaluate_model(model, test_df, feature_cols)

    print("\n=== Evaluation Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
