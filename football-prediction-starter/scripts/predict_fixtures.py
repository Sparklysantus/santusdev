from pathlib import Path
import argparse
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH  # noqa: E402
from src.data_io import load_matches  # noqa: E402
from src.features import build_feature_table  # noqa: E402
from src.predict_utils import FEATURE_COLS, build_fixture_feature_row, confidence_level, parse_fixture_date  # noqa: E402
from src.train import fit_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict outcomes for many fixtures from a CSV file.")
    parser.add_argument(
        "--fixtures",
        default="data/fixtures/fixtures_to_predict.csv",
        help="CSV input path with columns: home_team, away_team, fixture_date(optional)",
    )
    parser.add_argument(
        "--output",
        default="data/predictions/predictions.csv",
        help="CSV output path for predicted probabilities",
    )
    args = parser.parse_args()

    fixtures_path = Path(args.fixtures)
    if not fixtures_path.exists():
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")

    fixtures = pd.read_csv(fixtures_path)
    required = {"home_team", "away_team"}
    missing = required - set(fixtures.columns)
    if missing:
        raise ValueError(f"Missing required fixture columns: {sorted(missing)}")

    data_path = PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else RAW_DATA_PATH
    matches = load_matches(data_path)
    feature_df = build_feature_table(matches)
    model, _ = fit_model(feature_df)

    fallback_date = matches["date"].max()
    classes = model.named_steps["clf"].classes_
    predictions = []

    for _, row in fixtures.iterrows():
        fixture_date = parse_fixture_date(str(row["fixture_date"]) if "fixture_date" in fixtures.columns and pd.notna(row["fixture_date"]) else None, fallback_date)
        feat_row = build_fixture_feature_row(matches, str(row["home_team"]), str(row["away_team"]), fixture_date)
        probs = model.predict_proba(feat_row[FEATURE_COLS])[0]
        p_map = dict(zip(classes, probs))
        confidence = confidence_level(p_map)

        predictions.append(
            {
                "fixture_date": fixture_date.date().isoformat(),
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "prob_home_win": float(p_map.get("H", 0.0)),
                "prob_draw": float(p_map.get("D", 0.0)),
                "prob_away_win": float(p_map.get("A", 0.0)),
                "predicted_result": _argmax_label(p_map),
                "confidence_level": confidence,
            }
        )

    out_df = pd.DataFrame(predictions)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved predictions: {out_path}")
    print(f"Fixtures predicted: {len(out_df)}")


def _argmax_label(probability_map: dict[str, float]) -> str:
    label = max(probability_map.items(), key=lambda kv: kv[1])[0]
    return {"H": "Home Win", "D": "Draw", "A": "Away Win"}.get(label, label)


if __name__ == "__main__":
    main()
