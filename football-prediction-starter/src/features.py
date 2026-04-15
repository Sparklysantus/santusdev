import numpy as np
import pandas as pd


def build_feature_table(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy().sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index
    df["goal_diff"] = df["home_goals"] - df["away_goals"]

    long_df = _to_team_long(df)
    long_df = _add_rolling_form(long_df)
    long_df = _add_rolling_xg(long_df)
    long_df = _add_rest_days(long_df)

    home_feats = _extract_side_features(long_df, "home")
    away_feats = _extract_side_features(long_df, "away")

    merged = df.merge(home_feats, on=["match_id"], how="left")
    merged = merged.merge(away_feats, on=["match_id"], how="left")
    merged = _add_elo_features(merged)

    merged["form_points_diff"] = merged["home_form_points_5"] - merged["away_form_points_5"]
    merged["form_gd_diff"] = merged["home_form_goal_diff_5"] - merged["away_form_goal_diff_5"]
    merged["xg_diff"] = merged["home_form_xg_5"] - merged["away_form_xg_5"]
    merged["rest_days_diff"] = merged["home_rest_days"] - merged["away_rest_days"]
    merged["is_home"] = 1.0

    feature_cols = [
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

    for col in feature_cols:
        merged[col] = merged[col].fillna(0.0).astype(float)

    return merged


def _to_team_long(df: pd.DataFrame) -> pd.DataFrame:
    if "match_id" in df.columns:
        temp = df.copy()
    else:
        temp = df.reset_index().rename(columns={"index": "match_id"})

    home = temp[["match_id", "date", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    home.columns = ["match_id", "date", "team", "opponent", "goals_for", "goals_against"]
    home["side"] = "home"

    away = temp[["match_id", "date", "away_team", "home_team", "away_goals", "home_goals"]].copy()
    away.columns = ["match_id", "date", "team", "opponent", "goals_for", "goals_against"]
    away["side"] = "away"

    if "home_xg" in temp.columns and "away_xg" in temp.columns:
        home["xg_for"] = temp["home_xg"]
        home["xg_against"] = temp["away_xg"]
        away["xg_for"] = temp["away_xg"]
        away["xg_against"] = temp["home_xg"]
    else:
        home["xg_for"] = np.nan
        home["xg_against"] = np.nan
        away["xg_for"] = np.nan
        away["xg_against"] = np.nan

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["points"] = np.select(
        [long_df["goals_for"] > long_df["goals_against"], long_df["goals_for"] == long_df["goals_against"]],
        [3, 1],
        default=0,
    )
    long_df["goal_diff_team"] = long_df["goals_for"] - long_df["goals_against"]
    return long_df.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


def _add_rolling_form(long_df: pd.DataFrame) -> pd.DataFrame:
    group = long_df.groupby("team", group_keys=False)
    long_df["form_points_5"] = group["points"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    long_df["form_goal_diff_5"] = group["goal_diff_team"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    return long_df


def _add_rolling_xg(long_df: pd.DataFrame) -> pd.DataFrame:
    if "xg_for" not in long_df.columns:
        long_df["form_xg_5"] = np.nan
        return long_df

    group = long_df.groupby("team", group_keys=False)
    long_df["form_xg_5"] = group["xg_for"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    return long_df


def _add_rest_days(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df["prev_date"] = long_df.groupby("team")["date"].shift(1)
    long_df["rest_days"] = (long_df["date"] - long_df["prev_date"]).dt.days
    long_df["rest_days"] = long_df["rest_days"].fillna(7).clip(lower=1, upper=21)
    return long_df


def _extract_side_features(long_df: pd.DataFrame, side: str) -> pd.DataFrame:
    side_df = long_df[long_df["side"] == side][["match_id", "form_points_5", "form_goal_diff_5", "form_xg_5", "rest_days"]].copy()
    side_df.columns = [
        "match_id",
        f"{side}_form_points_5",
        f"{side}_form_goal_diff_5",
        f"{side}_form_xg_5",
        f"{side}_rest_days",
    ]
    return side_df


def _add_elo_features(df: pd.DataFrame, base_elo: float = 1500.0, k: float = 20.0) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    home_elo_list = []
    away_elo_list = []

    for _, row in df.sort_values("date").iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_elo = ratings.get(home_team, base_elo)
        away_elo = ratings.get(away_team, base_elo)

        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)

        expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400))
        if row["home_goals"] > row["away_goals"]:
            score_home = 1.0
        elif row["home_goals"] == row["away_goals"]:
            score_home = 0.5
        else:
            score_home = 0.0

        ratings[home_team] = home_elo + k * (score_home - expected_home)
        ratings[away_team] = away_elo + k * ((1.0 - score_home) - (1.0 - expected_home))

    out = df.copy().sort_values("date").reset_index(drop=True)
    out["home_elo_pre"] = home_elo_list
    out["away_elo_pre"] = away_elo_list
    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    return out
