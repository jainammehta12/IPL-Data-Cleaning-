from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

MATCHES_FILE = DATA_DIR / "matches.csv"
DELIVERIES_FILE = DATA_DIR / "deliveries.csv"


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find: {path}\n"
            "Make sure the CSV files are inside the project's data folder "
            "and are named matches.csv and deliveries.csv."
        )


def save_current_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def main():
    require_file(MATCHES_FILE)
    require_file(DELIVERIES_FILE)

    # 1. Load data
    matches = pd.read_csv(MATCHES_FILE)
    deliveries = pd.read_csv(DELIVERIES_FILE)

    print("=" * 65)
    print("IPL DATA CLEANING AND ANALYSIS")
    print("=" * 65)
    print(f"\nMatches dataset shape: {matches.shape}")
    print(f"Deliveries dataset shape: {deliveries.shape}")

    print("\nFirst five matches:")
    print(matches.head().to_string(index=False))
    print("\nFirst five deliveries:")
    print(deliveries.head().to_string(index=False))

    # 2. Missing values
    print("\n--- MISSING VALUES BEFORE CLEANING ---")
    print("\nMatches:")
    print(matches.isna().sum()[matches.isna().sum() > 0].to_string() or "No missing values")
    print("\nDeliveries:")
    print(deliveries.isna().sum()[deliveries.isna().sum() > 0].to_string() or "No missing values")

    # 3. Work on copies and standardize text
    matches_clean = matches.copy()
    deliveries_clean = deliveries.copy()

    for df in (matches_clean, deliveries_clean):
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    # Convert relevant types safely
    if "date" in matches_clean.columns:
        matches_clean["date"] = pd.to_datetime(matches_clean["date"], errors="coerce")

    for col in ("season", "result_margin"):
        if col in matches_clean.columns:
            matches_clean[col] = pd.to_numeric(matches_clean[col], errors="coerce")

    numeric_delivery_cols = (
        "match_id", "id", "inning", "over", "ball", "batsman_runs",
        "batter_runs", "extra_runs", "total_runs", "is_super_over"
    )
    for col in numeric_delivery_cols:
        if col in deliveries_clean.columns:
            deliveries_clean[col] = pd.to_numeric(deliveries_clean[col], errors="coerce")

    print("\n--- DATA TYPES AFTER CHECKING ---")
    print("\nMatches:")
    print(matches_clean.dtypes.to_string())
    print("\nDeliveries:")
    print(deliveries_clean.dtypes.to_string())

    # 4. Exact duplicates only
    match_duplicates = int(matches_clean.duplicated().sum())
    delivery_duplicates = int(deliveries_clean.duplicated().sum())
    print("\n--- DUPLICATE ROWS ---")
    print(f"Exact duplicate rows in matches: {match_duplicates}")
    print(f"Exact duplicate rows in deliveries: {delivery_duplicates}")

    matches_clean = matches_clean.drop_duplicates().copy()
    deliveries_clean = deliveries_clean.drop_duplicates().copy()

    print("\n--- DATASET SIZES AFTER REMOVING EXACT DUPLICATES ---")
    print(f"Matches: {matches_clean.shape}")
    print(f"Deliveries: {deliveries_clean.shape}")

    # 5. Potential outliers using IQR; report but do not automatically delete valid results.
    print("\n--- POTENTIAL OUTLIERS ---")
    if "result_margin" in matches_clean.columns:
        margin = pd.to_numeric(matches_clean["result_margin"], errors="coerce").dropna()
        if not margin.empty:
            q1, q3 = margin.quantile(0.25), margin.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = margin[(margin < lower) | (margin > upper)]
            print(f"Result-margin potential outliers by IQR: {len(outliers)}")
            print("These are flagged for investigation, not automatically removed.")
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=margin)
            plt.title("Potential Outliers in Match Result Margin")
            plt.xlabel("Result margin")
            save_current_plot("outliers_result_margin.png")
        else:
            print("No non-missing result margins are available.")
    else:
        print("The result_margin column is not present in this dataset.")

    # 6. Visualization 1: matches per season
    if "season" in matches_clean.columns:
        season_counts = matches_clean.dropna(subset=["season"]).groupby("season").size().sort_index()
        if not season_counts.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(season_counts.index, season_counts.values, marker="o")
            plt.title("IPL Matches Played per Season")
            plt.xlabel("Season")
            plt.ylabel("Number of matches")
            plt.xticks(season_counts.index, rotation=45)
            save_current_plot("matches_per_season.png")
            print("\nMatches per season:")
            print(season_counts.to_string())

    # 7. Visualization 2: team wins
    if "winner" in matches_clean.columns:
        wins = matches_clean["winner"].dropna().value_counts().head(10).sort_values()
        if not wins.empty:
            plt.figure(figsize=(10, 6))
            wins.plot(kind="barh")
            plt.title("Top Teams by IPL Match Wins")
            plt.xlabel("Matches won")
            plt.ylabel("Team")
            save_current_plot("matches_won_by_teams.png")
            print(f"\nMost wins in this dataset: {wins.idxmax()} ({int(wins.max())} wins)")

    # 8. Visualization 3: toss decisions
    if "toss_decision" in matches_clean.columns:
        toss_counts = matches_clean["toss_decision"].dropna().value_counts()
        if not toss_counts.empty:
            plt.figure(figsize=(7, 5))
            plt.pie(toss_counts.values, labels=toss_counts.index, autopct="%1.1f%%", startangle=90)
            plt.title("IPL Toss Decision Distribution")
            save_current_plot("toss_decision.png")

    # 9. Visualization 4: toss winner vs match winner
    if {"toss_winner", "winner"}.issubset(matches_clean.columns):
        decided = matches_clean.dropna(subset=["toss_winner", "winner"]).copy()
        if not decided.empty:
            decided["same_team"] = decided["toss_winner"] == decided["winner"]
            outcomes = decided["same_team"].map({
                True: "Toss winner also won",
                False: "Toss winner lost"
            }).value_counts()
            plt.figure(figsize=(8, 5))
            outcomes.plot(kind="bar", rot=0)
            plt.title("Did the Toss Winner Also Win the Match?")
            plt.xlabel("Outcome")
            plt.ylabel("Number of matches")
            save_current_plot("toss_vs_match_winner.png")
            print(f"Toss winner also won {100 * decided['same_team'].mean():.2f}% of matches with both values recorded.")

    # 10. Visualization 5: Player of the Match
    if "player_of_match" in matches_clean.columns:
        awards = matches_clean["player_of_match"].dropna().value_counts().head(10).sort_values()
        if not awards.empty:
            plt.figure(figsize=(10, 6))
            awards.plot(kind="barh")
            plt.title("Top 10 Player of the Match Awards")
            plt.xlabel("Awards")
            plt.ylabel("Player")
            save_current_plot("top_player_of_match.png")
            print(f"Most Player of the Match awards: {awards.idxmax()} ({int(awards.max())} awards)")

    # 11. Visualization 6: run scorers
    batter_col = "batter" if "batter" in deliveries_clean.columns else (
        "batsman" if "batsman" in deliveries_clean.columns else None
    )
    runs_col = "batsman_runs" if "batsman_runs" in deliveries_clean.columns else (
        "batter_runs" if "batter_runs" in deliveries_clean.columns else None
    )
    if batter_col and runs_col:
        run_totals = deliveries_clean.groupby(batter_col)[runs_col].sum().sort_values(ascending=False).head(10).sort_values()
        if not run_totals.empty:
            plt.figure(figsize=(10, 6))
            run_totals.plot(kind="barh")
            plt.title("Top 10 IPL Run Scorers")
            plt.xlabel("Runs")
            plt.ylabel("Batter")
            save_current_plot("top_run_scorers.png")
            print(f"Top run scorer: {run_totals.idxmax()} ({int(run_totals.max())} runs)")
    else:
        print("Could not identify batter/runs columns. Check the deliveries CSV column names.")

    # 12. Visualization 7: wickets credited to bowlers
    if {"bowler", "player_dismissed"}.issubset(deliveries_clean.columns):
        wicket_data = deliveries_clean[deliveries_clean["player_dismissed"].notna()].copy()
        if "dismissal_kind" in wicket_data.columns:
            not_credited = {"run out", "retired hurt", "retired out", "obstructing the field"}
            wicket_data = wicket_data[
                ~wicket_data["dismissal_kind"].astype(str).str.lower().isin(not_credited)
            ]
        wickets = wicket_data["bowler"].value_counts().head(10).sort_values()
        if not wickets.empty:
            plt.figure(figsize=(10, 6))
            wickets.plot(kind="barh")
            plt.title("Top 10 Bowlers by Wickets Credited")
            plt.xlabel("Wickets")
            plt.ylabel("Bowler")
            save_current_plot("top_wicket_takers.png")
            print(f"Most wickets credited: {wickets.idxmax()} ({int(wickets.max())} wickets)")

    # 13. Visualization 8: cities
    if "city" in matches_clean.columns:
        cities = matches_clean["city"].dropna().value_counts().head(10).sort_values()
        if not cities.empty:
            plt.figure(figsize=(10, 6))
            cities.plot(kind="barh")
            plt.title("Top 10 Cities by IPL Matches Hosted")
            plt.xlabel("Matches")
            plt.ylabel("City")
            save_current_plot("matches_by_city.png")

    # 14. Data quality summary
    summary = pd.DataFrame({
        "Dataset": ["matches.csv", "deliveries.csv"],
        "Rows before cleaning": [len(matches), len(deliveries)],
        "Rows after exact duplicate removal": [len(matches_clean), len(deliveries_clean)],
        "Exact duplicates removed": [len(matches) - len(matches_clean), len(deliveries) - len(deliveries_clean)],
        "Missing cells after cleaning": [
            int(matches_clean.isna().sum().sum()),
            int(deliveries_clean.isna().sum().sum())
        ]
    })
    print("\n--- DATA QUALITY SUMMARY ---")
    print(summary.to_string(index=False))

    # Save cleaned files for review
    matches_clean.to_csv(OUTPUT_DIR / "matches_cleaned.csv", index=False)
    deliveries_clean.to_csv(OUTPUT_DIR / "deliveries_cleaned.csv", index=False)

    print("\nFinished!")
    print(f"Charts and cleaned CSV files have been saved in: {OUTPUT_DIR}")
    print("Review the charts and printed results before writing your final conclusion.")


if __name__ == "__main__":
    main()

