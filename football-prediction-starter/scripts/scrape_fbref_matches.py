from pathlib import Path
import argparse
import io
import re
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FBREF_RAW_PATH  # noqa: E402


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def scrape_fbref_schedule(
    url: str | None = None,
    league_name: str = "UNKNOWN",
    html_text: str | None = None,
) -> pd.DataFrame:
    if html_text is None:
        if not url:
            raise ValueError("Either url or html_text must be provided.")
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        html_text = response.text

    soup = BeautifulSoup(html_text, "lxml")

    table_html = _extract_schedule_table_html(soup)
    if not table_html:
        raise ValueError("Could not find FBref schedule table on this page.")

    tables = pd.read_html(io.StringIO(table_html))
    if not tables:
        raise ValueError("No parseable schedule table found on FBref page.")

    raw = tables[0].copy()
    norm = {col: _normalize_col(col) for col in raw.columns}
    raw = raw.rename(columns=norm)

    required = {"date", "home", "away", "score"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing expected columns in FBref table: {sorted(missing)}")

    df = raw[["date", "home", "away", "score"]].copy()
    df = df[df["score"].astype(str).str.contains(r"\d+\D+\d+", regex=True, na=False)].copy()

    goals = df["score"].astype(str).str.extract(r"(?P<home_goals>\d+)\D+(?P<away_goals>\d+)")
    df["home_goals"] = pd.to_numeric(goals["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(goals["away_goals"], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"] = df["home"].astype(str).str.strip()
    df["away_team"] = df["away"].astype(str).str.strip()
    df["league"] = league_name

    out = df[["date", "league", "home_team", "away_team", "home_goals", "away_goals"]].dropna()
    out = out.sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def _extract_schedule_table_html(soup: BeautifulSoup) -> str:
    # Primary target: normal schedule table visible in DOM.
    table = soup.find("table", id="sched_all") or soup.find("table", id="sched")
    if not table:
        # Season pages often use dynamic ids like sched_2025-2026_9_1.
        table = soup.find("table", id=re.compile(r"^sched_"))
    if table:
        return str(table)

    # FBref often wraps full tables in HTML comments; search commented blocks.
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if "id=\"sched_all\"" not in comment and "id=\"sched\"" not in comment:
            continue
        inner = BeautifulSoup(comment, "lxml")
        comment_table = inner.find("table", id="sched_all") or inner.find("table", id="sched")
        if not comment_table:
            comment_table = inner.find("table", id=re.compile(r"^sched_"))
        if comment_table:
            return str(comment_table)
    return ""


def _normalize_col(col: object) -> str:
    text = str(col).strip().lower()
    text = re.sub(r"\s+", "_", text)
    mapping = {
        "home": "home",
        "away": "away",
        "date": "date",
        "score": "score",
    }
    return mapping.get(text, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape FBref season schedule into model-ready raw CSV.")
    parser.add_argument("--url", default=None, help="FBref competition season URL (schedule/results page).")
    parser.add_argument(
        "--html-file",
        default=None,
        help="Optional local saved FBref HTML file path (bypasses request blocking).",
    )
    parser.add_argument("--league", default="UNKNOWN", help="League label to store in output rows.")
    parser.add_argument("--output", default=str(FBREF_RAW_PATH), help="Output CSV path.")
    args = parser.parse_args()

    html_text = None
    if args.html_file:
        html_text = Path(args.html_file).read_text(encoding="utf-8", errors="ignore")

    try:
        df = scrape_fbref_schedule(url=args.url, league_name=args.league, html_text=html_text)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            raise SystemExit(
                "FBref blocked this automated request (HTTP 403).\n"
                "Open the FBref page in your browser, save page source to a local .html file, then run:\n"
                "python scripts/scrape_fbref_matches.py --html-file \"path/to/saved_fbref_page.html\" --league \"EPL\""
            ) from exc
        raise

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
