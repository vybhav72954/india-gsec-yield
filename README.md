# India 10Y G-Sec Univariate Time Series

Univariate analysis and forecasting of the daily India 10-year Government
Security yield (Bloomberg via FRTL Lab, 2010-01-01 → 2026-05-21, n = 4,275
weekday observations after Saturday-carry-forward cleanup).

The project follows the Sessions 1–4 forecasting + seasonality + breaks +
outliers syllabus, with a brief Appendix A on conditional-variance modelling.

## Notebooks

| File | What it is |
|---|---|
| `Univariate_India_10Y_Yield_v2.ipynb` | **The submission.** 27 main-body sections + 6 suffix subsections (6b HEGY, 8b break-aware unit roots, 9b SARIMA, 11b fixed-scheme robustness, 17b descriptive shock list, 19b empirical PI) + Appendix A (GARCH excursion). Runs end-to-end on the cleaned Monday–Friday series. |

Earlier iterations (the pre-appendix full notebook, the lightweight Sessions 1–2
version, and the notebook generator) are kept locally under `archive/` and are
not part of the published repository.

## Scripts

| File | Purpose |
|---|---|
| `build_clean_data.py` | Cleans the raw vendor parquet files into `clean_data/india_10y_weekday.csv` and related panel CSVs. Removes Saturday carry-forwards. |

The signal-location and standalone-workflow diagnostics that earlier lived in
separate scripts are now reproduced inside the notebook (Section 22 and the
main body) on the cleaned series; the original scripts are kept under `archive/`.

## Setup

```
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

Then either:
```
python build_clean_data.py      # rebuild cleaned CSVs
jupyter notebook Univariate_India_10Y_Yield_v2.ipynb
```

The notebook regenerates the cleaned data on demand if `clean_data/` is missing.

## Headline result

ARIMA(0,1,3) on the yield level is statistically indistinguishable from the
random walk at every forecast horizon under both Recursive and Fixed
estimation schemes. The I(1) finding survives all five break-aware
unit-root tests; HEGY rules out seasonal unit roots; Chen-Liu detects ~30
formal outliers concentrated on the 2013 RBI MSF window. Residual ARCH-LM
rejects conditional homoscedasticity — the volatility excursion in
Appendix A shows that a GARCH-conditional 95% PI reaches nominal coverage
(95.6%, p = 0.53) at about 37% narrower width than the constant-σ² ARIMA
PI.
