"""
Build cleaned datasets for the yield-curve project.

The raw wide panel mixes calendars and frequencies:
  - Indian yields include carried-forward Saturdays.
  - US yields are Monday-Friday.
  - Macro variables are lower frequency and forward-filled.

Outputs are CSV-first so they can be inspected and reused directly.
"""

from pathlib import Path

import pandas as pd


RAW_10Y = Path("yield_curve_data/cache/IND_10Y_daily.parquet")
RAW_WIDE = Path("yield_curve_data/yield_curve_daily_wide.parquet")
CACHE_DIR = Path("yield_curve_data/cache")
OUT_DIR = Path("clean_data")
SERIES_CSV_DIR = OUT_DIR / "series_csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SERIES_CSV_DIR.mkdir(parents=True, exist_ok=True)

IND_YIELD_PREFIXES = ("IND_1Y", "IND_2Y", "IND_5Y", "IND_7Y", "IND_10Y", "IND_30Y")
US_YIELD_PREFIXES = ("US_3M", "US_2Y", "US_5Y", "US_10Y", "US_30Y")


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            df = df.set_index(date_cols[0])
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def series_name_from_path(path: Path) -> str:
    return path.stem.removesuffix("_daily")


def first_value_column(df: pd.DataFrame) -> str:
    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["yield", "px_last", "close", "value", "last"])
    ]
    return candidates[0] if candidates else df.columns[0]


def clean_single_cache_file(path: Path) -> tuple[pd.DataFrame, dict]:
    name = series_name_from_path(path)
    raw = ensure_datetime_index(pd.read_parquet(path))
    value_col = first_value_column(raw)
    s = raw[[value_col]].rename(columns={value_col: name}).sort_index()

    raw_rows = len(s)
    raw_nan = int(s[name].isna().sum())
    raw_saturdays = int((s.index.dayofweek == 5).sum())
    raw_sundays = int((s.index.dayofweek == 6).sum())

    # All modeling datasets use a Monday-Friday grid. This removes the
    # carried-forward Saturdays in Indian yield files and aligns US yields.
    clean = s[s.index.dayofweek < 5].dropna().copy()

    out_csv = SERIES_CSV_DIR / f"{name}_weekday.csv"
    clean.to_csv(out_csv, index_label="date")

    audit = {
        "series": name,
        "source_file": str(path),
        "raw_rows": raw_rows,
        "raw_nan": raw_nan,
        "raw_saturdays": raw_saturdays,
        "raw_sundays": raw_sundays,
        "clean_rows": len(clean),
        "clean_nan": int(clean[name].isna().sum()),
        "clean_start": clean.index.min().date().isoformat() if len(clean) else "",
        "clean_end": clean.index.max().date().isoformat() if len(clean) else "",
        "csv_file": str(out_csv),
    }

    if name.startswith(IND_YIELD_PREFIXES):
        saturday = s[s.index.dayofweek == 5]
        audit["saturdays_equal_previous_observation"] = int(
            (saturday[name] == s[name].shift(1).reindex(saturday.index)).sum()
        )
    else:
        audit["saturdays_equal_previous_observation"] = ""

    return clean, audit


def clean_all_cache_series() -> dict[str, pd.DataFrame]:
    cleaned = {}
    audit_rows = []

    for path in sorted(CACHE_DIR.glob("*_daily.parquet")):
        clean, audit = clean_single_cache_file(path)
        cleaned[audit["series"]] = clean
        audit_rows.append(audit)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_DIR / "all_series_cleaning_audit.csv", index=False)
    return cleaned


def clean_india_10y() -> pd.DataFrame:
    raw = ensure_datetime_index(pd.read_parquet(RAW_10Y))
    col = raw.columns[0]
    y = raw[[col]].rename(columns={col: "yield"}).dropna()

    saturday = y[y.index.dayofweek == 5]
    saturday_equal_prev = int((saturday["yield"] == y["yield"].shift(1).reindex(saturday.index)).sum())

    clean = y[y.index.dayofweek < 5].copy()
    clean.to_parquet(OUT_DIR / "india_10y_weekday.parquet")
    clean.to_csv(OUT_DIR / "india_10y_weekday.csv", index_label="date")

    audit = pd.DataFrame(
        [
            {"metric": "raw_rows", "value": len(y)},
            {"metric": "raw_nan", "value": int(y["yield"].isna().sum())},
            {"metric": "raw_saturdays", "value": len(saturday)},
            {"metric": "saturdays_equal_previous_observation", "value": saturday_equal_prev},
            {"metric": "clean_rows_mon_fri", "value": len(clean)},
            {"metric": "clean_nan", "value": int(clean["yield"].isna().sum())},
            {"metric": "clean_start", "value": clean.index.min().date().isoformat()},
            {"metric": "clean_end", "value": clean.index.max().date().isoformat()},
        ]
    )
    audit.to_csv(OUT_DIR / "india_10y_cleaning_audit.csv", index=False)
    return clean


def clean_yield_curve_panel() -> pd.DataFrame:
    wide = ensure_datetime_index(pd.read_parquet(RAW_WIDE))

    # Keep the panel on a common Monday-Friday daily grid. Do not impute here:
    # model-specific scripts can choose inner joins, forward-filled macro data,
    # or lower-frequency aggregation explicitly.
    panel = wide[wide.index.dayofweek < 5].copy()
    panel.to_parquet(OUT_DIR / "yield_curve_panel_weekday.parquet")
    panel.to_csv(OUT_DIR / "yield_curve_panel_weekday.csv", index_label="date")

    indian_curve_cols = [c for c in panel.columns if c.startswith(IND_YIELD_PREFIXES)]
    us_curve_cols = [c for c in panel.columns if c.startswith(US_YIELD_PREFIXES)]
    if indian_curve_cols:
        panel[indian_curve_cols].dropna().to_csv(
            OUT_DIR / "india_yield_curve_weekday_complete.csv", index_label="date"
        )
    if us_curve_cols:
        panel[us_curve_cols].dropna().to_csv(
            OUT_DIR / "us_yield_curve_weekday_complete.csv", index_label="date"
        )
    if indian_curve_cols and us_curve_cols:
        both_cols = indian_curve_cols + us_curve_cols
        panel[both_cols].dropna().to_csv(
            OUT_DIR / "india_us_yield_curve_weekday_complete.csv", index_label="date"
        )

    missing = (
        panel.isna()
        .agg(["sum", "mean"])
        .T.rename(columns={"sum": "missing_count", "mean": "missing_rate"})
    )
    missing["missing_rate"] = missing["missing_rate"].round(4)
    missing.to_csv(OUT_DIR / "yield_curve_panel_missing_audit.csv")
    return panel


if __name__ == "__main__":
    cleaned = clean_all_cache_series()
    y10 = clean_india_10y()
    panel = clean_yield_curve_panel()

    print("Clean India 10Y weekday series")
    print(f"  rows : {len(y10)}")
    print(f"  span : {y10.index.min().date()} -> {y10.index.max().date()}")
    print(f"  csv  : {OUT_DIR / 'india_10y_weekday.csv'}")

    print("\nClean weekday yield-curve panel")
    print(f"  shape: {panel.shape}")
    print(f"  span : {panel.index.min().date()} -> {panel.index.max().date()}")
    print(f"  csv  : {OUT_DIR / 'yield_curve_panel_weekday.csv'}")

    print("\nClean individual series CSVs")
    print(f"  count: {len(cleaned)}")
    print(f"  dir  : {SERIES_CSV_DIR}")
