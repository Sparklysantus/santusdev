import numpy as np
import pandas as pd


FEATURE_COLS = [
    "home_form_points_5",
    "away_form_points_5",
    "form_points_diff",
    "home_form_goal_diff_5",
    "away_form_goal_diff_5",
    "form_gd_diff",
    "home_form_xg_5",
    "away_form_xg_5",
    "xg_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff",
    "is_home",
]


def parse_fixture_date(date_arg: str | None, fallback: pd.Timestamp) -> pd.Timestamp:
    if date_arg is None:
        return fallback + pd.Timedelta(days=7)
    return pd.to_datetime(date_arg, errors="raise")


def confidence_level(probability_map: dict[str, float]) -> int:
    """
    Convert model certainty to a 1-10 confidence score.
    Uses top predicted probability; 0.0->1, 1.0->10.
    """
    if not probability_map:
        return 1
    top_prob = max(probability_map.values())
    score = int(round(top_prob * 10))
    return max(1, min(10, score))


def build_fixture_feature_row(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    fixture_date: pd.Timestamp,
) -> pd.DataFrame:
    long_df = _to_long(matches)
    ratings = _elo_ratings_after_history(matches)

    home_form_pts, home_form_gd, home_form_xg = _recent_form(long_df, home_team)
    away_form_pts, away_form_gd, away_form_xg = _recent_form(long_df, away_team)
    home_rest = _rest_days(long_df, home_team, fixture_date)
    away_rest = _rest_days(long_df, away_team, fixture_date)
    home_elo = ratings.get(home_team, 1500.0)
    away_elo = ratings.get(away_team, 1500.0)

    data = {
        "home_form_points_5": home_form_pts,
        "away_form_points_5": away_form_pts,
        "form_points_diff": home_form_pts - away_form_pts,
        "home_form_goal_diff_5": home_form_gd,
        "away_form_goal_diff_5": away_form_gd,
        "form_gd_diff": home_form_gd - away_form_gd,
        "home_form_xg_5": home_form_xg,
        "away_form_xg_5": away_form_xg,
        "xg_diff": home_form_xg - away_form_xg,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_days_diff": home_rest - away_rest,
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "elo_diff": home_elo - away_elo,
        "is_home": 1.0,
    }
    return pd.DataFrame([data])


def _to_long(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches[["date", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    home.columns = ["date", "team", "opponent", "goals_for", "goals_against"]
    home["points"] = ((home["goals_for"] > home["goals_against"]) * 3 + (home["goals_for"] == home["goals_against"]) * 1).astype(float)
    home["goal_diff"] = home["goals_for"] - home["goals_against"]

    away = matches[["date", "away_team", "home_team", "away_goals", "home_goals"]].copy()
    away.columns = ["date", "team", "opponent", "goals_for", "goals_against"]
    away["points"] = ((away["goals_for"] > away["goals_against"]) * 3 + (away["goals_for"] == away["goals_against"]) * 1).astype(float)
    away["goal_diff"] = away["goals_for"] - away["goals_against"]

    if "home_xg" in matches.columns and "away_xg" in matches.columns:
        home["xg_for"] = matches["home_xg"]
        home["xg_against"] = matches["away_xg"]
        away["xg_for"] = matches["away_xg"]
        away["xg_against"] = matches["home_xg"]
    else:
        home["xg_for"] = np.nan
        home["xg_against"] = np.nan
        away["xg_for"] = np.nan
        away["xg_against"] = np.nan

    return pd.concat([home, away], ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)


def _recent_form(long_df: pd.DataFrame, team: str) -> tuple[float, float, float]:
    team_df = long_df[long_df["team"] == team].sort_values("date")
    if team_df.empty:
        return 0.0, 0.0, 0.0
    last5 = team_df.tail(5)
    points_mean = float(last5["points"].mean())
    goal_diff_mean = float(last5["goal_diff"].mean())
    xg_mean = 0.0
    if "xg_for" in last5.columns:
        xg_mean = float(last5["xg_for"].mean()) if not last5["xg_for"].empty else 0.0
        if not pd.notna(xg_mean):
            xg_mean = 0.0
    return points_mean, goal_diff_mean, xg_mean


def _rest_days(long_df: pd.DataFrame, team: str, fixture_date: pd.Timestamp) -> float:
    team_df = long_df[long_df["team"] == team].sort_values("date")
    if team_df.empty:
        return 7.0
    last_date = team_df["date"].iloc[-1]
    days = (fixture_date - last_date).days
    return float(max(1, min(21, days if pd.notna(days) else 7)))


def _elo_ratings_after_history(matches: pd.DataFrame, base_elo: float = 1500.0, k: float = 20.0) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for _, row in matches.sort_values("date").iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_elo = ratings.get(home_team, base_elo)
        away_elo = ratings.get(away_team, base_elo)

        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400))
        if row["home_goals"] > row["away_goals"]:
            score_home = 1.0
        elif row["home_goals"] == row["away_goals"]:
            score_home = 0.5
        else:
            score_home = 0.0

        ratings[home_team] = home_elo + k * (score_home - expected_home)
        ratings[away_team] = away_elo + k * ((1.0 - score_home) - (1.0 - expected_home))
    return ratings
