# Airbnb Statistical Analysis

Statistical analysis of Airbnb prices across major European cities using t-tests, ANOVA, correlation, and regression.

## Dataset
Source: https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities

The script expects CSV files under `Yeni klasor/` (or `Yeni klasor` with the Turkish character if your folder uses it).

## How to Run
1) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas scipy pingouin statsmodels matplotlib
```

2) Run the analysis:

```bash
python StatisticalAnalysis.py
```

## Outputs
The script prints statistical test results and generates plots such as:
- Price distributions and group comparisons
- Distance vs price scatter plots
- City-level comparisons
- Regression diagnostics and coefficient plots

Sample images are saved in the repository root (PNG files).

## Notes
- Prices are log-transformed for regression to reduce skew.
- City is treated as a categorical variable with a reference city.
