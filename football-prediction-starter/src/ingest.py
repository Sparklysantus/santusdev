from pathlib import Path
import re

import pandas as pd

from src.config import REQUIRED_COLUMNS


def load_fbref_matches(path: Path) -> pd.DataFrame:
    """
    Load an FBref-like match export and normalize it to the model schema.
    Expected raw columns (case-insensitive variants supported):
    - date, home_team, away_team, home_goals, away_goals
    Optional:
    - league
    - home_xg, away_xg
    """
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"No rows found in {path}")

    raw = _normalize_column_names(raw)
    for col in ["date", "home_team", "away_team", "home_goals", "away_goals"]:
        if col not in raw.columns:
            raise ValueError(f"Required column '{col}' not found in {path}")

    out = raw.copy()
    if "league" not in out.columns:
        out["league"] = "UNKNOWN"

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")

    if "home_xg" in out.columns:
        out["home_xg"] = pd.to_numeric(out["home_xg"], errors="coerce")
    if "away_xg" in out.columns:
        out["away_xg"] = pd.to_numeric(out["away_xg"], errors="coerce")

    out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    out["home_team"] = out["home_team"].map(_clean_team_name)
    out["away_team"] = out["away_team"].map(_clean_team_name)

    out_cols = ["date", "league", "home_team", "away_team", "home_goals", "away_goals"]
    for col in OPTIONAL_COLUMNS:
        if col in out.columns:
            out_cols.append(col)

    out = out[out_cols].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def apply_team_name_map(df: pd.DataFrame, map_path: Path) -> pd.DataFrame:
    """
    Map source team names to canonical names.
    Mapping file must include columns: source_name, canonical_name.
    """
    if not map_path.exists():
        return df

    name_map_df = pd.read_csv(map_path)
    if name_map_df.empty:
        return df

    if "source_name" not in name_map_df.columns or "canonical_name" not in name_map_df.columns:
        raise ValueError("team_name_map.csv must contain source_name and canonical_name columns.")

    map_dict = dict(zip(name_map_df["source_name"].astype(str), name_map_df["canonical_name"].astype(str)))
    out = df.copy()
    out["home_team"] = out["home_team"].map(lambda x: map_dict.get(x, x))
    out["away_team"] = out["away_team"].map(lambda x: map_dict.get(x, x))
    return out


def merge_team_context(matches_df: pd.DataFrame, context_path: Path) -> pd.DataFrame:
    """
    Merge optional team-day context (e.g., injuries, market value).
    Expected context schema:
    date,team,injury_count,market_value_eur
    """
    if not context_path.exists():
        return matches_df

    ctx = pd.read_csv(context_path)
    if ctx.empty:
        return matches_df

    required = {"date", "team"}
    missing = required - set(ctx.columns)
    if missing:
        raise ValueError(f"Missing context columns: {sorted(missing)}")

    ctx["date"] = pd.to_datetime(ctx["date"], errors="coerce")
    ctx["team"] = ctx["team"].map(_clean_team_name)

    if "injury_count" in ctx.columns:
        ctx["injury_count"] = pd.to_numeric(ctx["injury_count"], errors="coerce")
    if "market_value_eur" in ctx.columns:
        ctx["market_value_eur"] = pd.to_numeric(ctx["market_value_eur"], errors="coerce")

    home_ctx = ctx.rename(
        columns={
            "team": "home_team",
            "injury_count": "home_injury_count",
            "market_value_eur": "home_market_value_eur",
        }
    )
    away_ctx = ctx.rename(
        columns={
            "team": "away_team",
            "injury_count": "away_injury_count",
            "market_value_eur": "away_market_value_eur",
        }
    )

    out = matches_df.merge(
        home_ctx[["date", "home_team"] + _available_cols(home_ctx, ["home_injury_count", "home_market_value_eur"])],
        on=["date", "home_team"],
        how="left",
    )
    out = out.merge(
        away_ctx[["date", "away_team"] + _available_cols(away_ctx, ["away_injury_count", "away_market_value_eur"])],
        on=["date", "away_team"],
        how="left",
    )

    if "home_injury_count" in out.columns and "away_injury_count" in out.columns:
        out["injury_count_diff"] = out["home_injury_count"].fillna(0) - out["away_injury_count"].fillna(0)
    if "home_market_value_eur" in out.columns and "away_market_value_eur" in out.columns:
        out["market_value_diff_eur"] = out["home_market_value_eur"].fillna(0) - out["away_market_value_eur"].fillna(0)

    return out


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "date": "date",
        "competition": "league",
        "league": "league",
        "home": "home_team",
        "home_team": "home_team",
        "away": "away_team",
        "away_team": "away_team",
        "home_goals": "home_goals",
        "away_goals": "away_goals",
        "home_xg": "home_xg",
        "away_xg": "away_xg",
        "xg_home": "home_xg",
        "xg_away": "away_xg",
        "score": "score",
    }

    renamed = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        renamed[col] = alias_map.get(key, key)
    out = df.rename(columns=renamed)

    if "score" in out.columns and ("home_goals" not in out.columns or "away_goals" not in out.columns):
        parsed = out["score"].astype(str).str.extract(r"(?P<h>\d+)\D+(?P<a>\d+)")
        out["home_goals"] = pd.to_numeric(parsed["h"], errors="coerce")
        out["away_goals"] = pd.to_numeric(parsed["a"], errors="coerce")

    return out


def _clean_team_name(name: str) -> str:
    if pd.isna(name):
        return name
    cleaned = re.sub(r"\s+", " ", str(name)).strip()
    return cleaned


def _available_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]
