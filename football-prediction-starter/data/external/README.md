# External Data Drop Folder

Place third-party source exports here before building a model-ready dataset.

## Files

- `fbref_matches.csv` (required)
  - Minimum columns:
    - `date`
    - `home_team` (or `home`)
    - `away_team` (or `away`)
    - `home_goals` + `away_goals`, or a single `score` column like `2-1`
  - Optional: `league` or `competition`

- `transfermarkt_team_context.csv` (optional)
  - Recommended columns:
    - `date`
    - `team`
    - `injury_count`
    - `market_value_eur`

## Build processed dataset

From project root:

```bash
python scripts/build_dataset.py
```

This writes `data/processed/matches_model_ready.csv`.
