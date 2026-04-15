from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FBREF_RAW_PATH, PROCESSED_DATA_PATH, TEAM_NAME_MAP_PATH, TRANSFERMARKT_CONTEXT_PATH
from src.ingest import apply_team_name_map, load_fbref_matches, merge_team_context


def main() -> None:
    if not FBREF_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Expected FBref source at {FBREF_RAW_PATH}. "
            "Drop your FBref export there first."
        )

    matches = load_fbref_matches(FBREF_RAW_PATH)
    matches = apply_team_name_map(matches, TEAM_NAME_MAP_PATH)
    matches = merge_team_context(matches, TRANSFERMARKT_CONTEXT_PATH)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Built model-ready dataset: {PROCESSED_DATA_PATH}")
    print(f"Rows: {len(matches)}")
    print(f"Columns: {list(matches.columns)}")


if __name__ == "__main__":
    main()
