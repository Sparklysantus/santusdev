from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH  # noqa: E402
from src.data_io import load_matches  # noqa: E402
from src.features import build_feature_table  # noqa: E402
from src.predict_utils import FEATURE_COLS, build_fixture_feature_row, confidence_level, parse_fixture_date  # noqa: E402
from src.train import fit_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Home/Draw/Away probabilities for a fixture.")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument("--date", default=None, help="Fixture date YYYY-MM-DD (optional)")
    args = parser.parse_args()

    data_path = PROCESSED_DATA_PATH if PROCESSED_DATA_PATH.exists() else RAW_DATA_PATH
    matches = load_matches(data_path)
    feature_df = build_feature_table(matches)

    model, _ = fit_model(feature_df)
    fixture_date = parse_fixture_date(args.date, matches["date"].max())
    row = build_fixture_feature_row(matches, args.home, args.away, fixture_date)

    probs = model.predict_proba(row[FEATURE_COLS])[0]
    classes = model.named_steps["clf"].classes_
    probability_map = dict(zip(classes, probs))
    confidence = confidence_level(probability_map)

    print(f"Fixture: {args.home} vs {args.away} ({fixture_date.date()})")
    print("Predicted probabilities:")
    print(f"  Home Win (H): {probability_map.get('H', 0.0):.4f}")
    print(f"  Draw (D):     {probability_map.get('D', 0.0):.4f}")
    print(f"  Away Win (A): {probability_map.get('A', 0.0):.4f}")
    print(f"  Confidence:   {confidence}/10")


if __name__ == "__main__":
    main()
