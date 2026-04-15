import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_PATH, TARGET_COLUMN


def fit_model(train_df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
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

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, MODEL_PATH)
    return model, feature_cols
