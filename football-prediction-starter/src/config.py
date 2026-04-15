from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "matches_sample.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches_model_ready.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "logreg_match_outcome.joblib"

FBREF_RAW_PATH = PROJECT_ROOT / "data" / "external" / "fbref_matches.csv"
TRANSFERMARKT_CONTEXT_PATH = PROJECT_ROOT / "data" / "external" / "transfermarkt_team_context.csv"
TEAM_NAME_MAP_PATH = PROJECT_ROOT / "data" / "mappings" / "team_name_map.csv"

REQUIRED_COLUMNS = [
    "date",
    "league",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]

OPTIONAL_COLUMNS = [
    "home_xg",
    "away_xg",
]

TARGET_COLUMN = "result"
