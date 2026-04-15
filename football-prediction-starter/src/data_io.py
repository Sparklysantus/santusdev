import pandas as pd

from src.config import REQUIRED_COLUMNS, TARGET_COLUMN


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    df[TARGET_COLUMN] = df.apply(_result_label, axis=1)
    return df


def _result_label(row: pd.Series) -> str:
    if row["home_goals"] > row["away_goals"]:
        return "H"
    if row["home_goals"] < row["away_goals"]:
        return "A"
    return "D"


def time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1.")

    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty partition. Add more rows.")

    return train_df, test_df
