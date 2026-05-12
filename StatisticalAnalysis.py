from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt


def format_p_value(p_value: float, log10_p: Optional[float] = None) -> str:
	if p_value is None or not np.isfinite(p_value):
		return "nan"
	if p_value == 0:
		if log10_p is not None and np.isfinite(log10_p):
			return f"< 1e-300 (log10 p ~= {log10_p:.2f})"
		return "< 1e-300"
	if p_value < 1e-6:
		return f"{p_value:.3e}"
	return f"{p_value:.6g}"


def print_header(title: str, lines: Optional[list[str]] = None, width: int = 70) -> None:
	print("=" * width)
	print(title)
	if lines:
		for line in lines:
			print(line)
	print("=" * width)


def print_decision(p_value: float, alpha: float, reject_msg: str, fail_msg: str) -> None:
	if p_value < alpha:
		print("Decision: Reject H0")
		if reject_msg:
			print(f"Interpretation: {reject_msg}")
	else:
		print("Decision: Fail to reject H0")
		if fail_msg:
			print(f"Interpretation: {fail_msg}")


def extract_p_value(result_df: pd.DataFrame) -> float:
	for col in ("p-unc", "p-value", "p"):
		if col in result_df.columns:
			return result_df[col].values[0]
	p_cols = [col for col in result_df.columns if "p" in col.lower()]
	return result_df[p_cols[0]].values[0] if p_cols else np.nan


def parse_city_day(stem: str) -> Tuple[str, str]:
	parts = stem.split("_")
	if len(parts) < 2:
		return stem.title(), "unknown"
	city = "_".join(parts[:-1]).title()
	day_type = parts[-1].lower()
	return city, day_type


def load_merged_data(data_dir: Path) -> pd.DataFrame:
	csv_paths = sorted(data_dir.glob("*_week*.csv"))
	if not csv_paths:
		raise FileNotFoundError("No '*_week*.csv' files found under 'Yeni klasor'.")

	frames = []
	for csv_path in csv_paths:
		temp = pd.read_csv(csv_path)
		unnamed_cols = [col for col in temp.columns if col.startswith("Unnamed")]
		if unnamed_cols:
			temp = temp.drop(columns=unnamed_cols)

		city, day_type = parse_city_day(csv_path.stem)
		temp["city"] = city
		temp["day_type"] = day_type
		frames.append(temp)

	df = pd.concat(frames, ignore_index=True)
	if "realSum" in df.columns and "price" not in df.columns:
		df = df.rename(columns={"realSum": "price"})

	return df


def run_city_anova(df: pd.DataFrame, value_col: str, title: str, hypothesis_label: str, alpha: float = 0.05) -> None:
	if "city" not in df.columns:
		print("Warning: 'city' column not found in dataset.")
		return

	if value_col not in df.columns:
		print(f"Warning: '{value_col}' column not found in dataset.")
		return

	df_clean = df[["city", value_col]].dropna()
	cities = sorted([city for city in df_clean["city"].unique() if pd.notna(city)])

	if len(cities) < 2:
		print(f"Warning: Only {len(cities)} city(ies) found; ANOVA requires >= 2 groups.")
		return

	groups = []
	group_stats = []
	for city in cities:
		values = df_clean.loc[df_clean["city"] == city, value_col].values
		if len(values) > 0:
			groups.append(values)
			group_stats.append({
				"city": city,
				"n": len(values),
				"mean": np.mean(values),
				"median": np.median(values),
				"std": np.std(values, ddof=1),
				"var": np.var(values, ddof=1),
			})

	if len(groups) < 2:
		print("Warning: Not enough non-empty city groups for ANOVA.")
		return

	n_total = sum(len(g) for g in groups)
	df_between = len(groups) - 1
	df_within = n_total - len(groups)

	levene_stat, levene_p = stats.levene(*groups)
	levene_log10 = None
	if df_within > 0:
		levene_logsf = stats.f.logsf(levene_stat, df_between, df_within)
		if np.isfinite(levene_logsf):
			levene_log10 = levene_logsf / np.log(10)
	use_welch = levene_p < alpha

	if use_welch:
		df_pg = df_clean.rename(columns={"city": "group", value_col: "value"})
		welch = pg.welch_anova(dv="value", between="group", data=df_pg)
		f_stat = welch["F"].values[0]
		p_log10 = None
		p_value = extract_p_value(welch)
		if "ddof1" in welch.columns and "ddof2" in welch.columns:
			welch_logsf = stats.f.logsf(f_stat, welch["ddof1"].values[0], welch["ddof2"].values[0])
			if np.isfinite(welch_logsf):
				p_log10 = welch_logsf / np.log(10)
		test_name = "Welch ANOVA"
	else:
		f_stat, p_value = stats.f_oneway(*groups)
		p_log10 = None
		if df_within > 0:
			anova_logsf = stats.f.logsf(f_stat, df_between, df_within)
			if np.isfinite(anova_logsf):
				p_log10 = anova_logsf / np.log(10)
		test_name = "One-way ANOVA"

	grand_mean = df_clean[value_col].mean()
	ss_total = np.sum((df_clean[value_col] - grand_mean) ** 2)
	ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
	eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

	print_header(title, [hypothesis_label], width=80)
	for stat_row in group_stats:
		print(f"{stat_row['city']:20s} -> n={stat_row['n']:6,}, mean={stat_row['mean']:8.3f}, median={stat_row['median']:8.3f}, std={stat_row['std']:8.3f}, var={stat_row['var']:8.3f}")
	print("-" * 80)
	print(f"Levene test p-value: {format_p_value(levene_p, levene_log10)}")
	print(f"Selected test: {test_name}")
	print(f"F-statistic: {f_stat:.4f}")
	print(f"p-value: {format_p_value(p_value, p_log10)}")
	print(f"Eta-squared: {eta_sq:.4f}")
	print("-" * 80)
	print_decision(
		p_value,
		alpha,
		reject_msg="At least one city mean differs.",
		fail_msg="No evidence of different city means.",
	)


def test_city_anova_price(df: pd.DataFrame, alpha: float = 0.05) -> None:
	run_city_anova(
		df=df,
		value_col="price",
		title="RQ3A: Effect of City on Price",
		hypothesis_label="H0: mu_city1 = mu_city2 = ... = mu_cityK\nH1: At least one city has a different mean price",
		alpha=alpha,
	)


def test_city_anova_satisfaction(df: pd.DataFrame, alpha: float = 0.05) -> None:
	run_city_anova(
		df=df,
		value_col="guest_satisfaction_overall",
		title="RQ3B: Effect of City on Guest Satisfaction",
		hypothesis_label="H0: mu_city1 = mu_city2 = ... = mu_cityK\nH1: At least one city has a different mean guest satisfaction",
		alpha=alpha,
	)


def test_weekday_vs_weekend_price(df: pd.DataFrame, alpha: float = 0.05) -> None:
	weekday = df.loc[df["day_type"] == "weekdays", "price"].dropna()
	weekend = df.loc[df["day_type"] == "weekends", "price"].dropna()

	if len(weekday) == 0 or len(weekend) == 0:
		raise ValueError("Weekday or weekend group is empty. Check day_type labels.")

	n1, n2 = len(weekday), len(weekend)
	mean1, mean2 = weekday.mean(), weekend.mean()
	med1, med2 = weekday.median(), weekend.median()
	std1, std2 = weekday.std(ddof=1), weekend.std(ddof=1)

	t_value, p_value = stats.ttest_ind(weekday, weekend, equal_var=False)

	var1, var2 = std1**2, std2**2
	se1, se2 = var1 / n1, var2 / n2
	df_num = (se1 + se2) ** 2
	df_den = (se1**2) / (n1 - 1) + (se2**2) / (n2 - 1)
	df_welch = df_num / df_den if df_den > 0 else np.nan
	t_critical = stats.t.ppf(1 - alpha / 2, df_welch) if np.isfinite(df_welch) else np.nan

	pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
	cohen_d = (mean2 - mean1) / pooled_sd if pooled_sd > 0 else np.nan

	print_header(
		"RQ1: Difference Between Weekday and Weekend Prices",
		["H0: mu_weekday = mu_weekend", "H1: mu_weekday != mu_weekend"],
	)
	print(f"Weekday  -> n={n1:,}, mean={mean1:.3f}, median={med1:.3f}, std={std1:.3f}")
	print(f"Weekend  -> n={n2:,}, mean={mean2:.3f}, median={med2:.3f}, std={std2:.3f}")
	print("-" * 70)
	print(f"Welch t value: {t_value:.4f}")
	print(f"t critical (two-tailed, alpha={alpha}): {t_critical:.4f}")
	print(f"p-value: {format_p_value(p_value)}")
	print(f"Welch df: {df_welch:.2f}")
	print(f"Cohen's d (weekend - weekday): {cohen_d:.4f}")
	print("-" * 70)
	print_decision(
		p_value,
		alpha,
		reject_msg="Weekday and weekend mean prices differ.",
		fail_msg="No significant mean difference detected.",
	)


def test_room_type_anova(df: pd.DataFrame, alpha: float = 0.05) -> None:
	if "room_type" not in df.columns:
		print("Warning: 'room_type' column not found.")
		return

	df_clean = df[["room_type", "price"]].dropna()

	room_types = sorted(df_clean["room_type"].unique())

	groups = []
	group_stats = []

	for rt in room_types:
		prices = df_clean.loc[df_clean["room_type"] == rt, "price"].values
		if len(prices) > 0:
			groups.append(prices)
			group_stats.append({
				"room_type": rt,
				"n": len(prices),
				"mean": prices.mean(),
				"median": np.median(prices),
				"std": np.std(prices, ddof=1),
			})

	levene_stat, levene_p = stats.levene(*groups)

	use_welch = levene_p < alpha

	if use_welch:
		df_pg = df_clean.rename(columns={"room_type": "group", "price": "value"})
		welch = pg.welch_anova(dv="value", between="group", data=df_pg)
		f_stat = welch["F"].values[0]
		p_value = extract_p_value(welch)
		test_name = "Welch ANOVA"
	else:
		f_stat, p_value = stats.f_oneway(*groups)
		test_name = "One-way ANOVA"

	grand_mean = df_clean["price"].mean()
	ss_total = np.sum((df_clean["price"] - grand_mean) ** 2)
	ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
	eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

	print_header(
		"RQ2: Effect of Room Type on Price",
		[
			"H0: mu_room_type1 = mu_room_type2 = ... = mu_room_typeK",
			"H1: At least one room type differs",
		],
	)

	for stat_row in group_stats:
		print(f"{stat_row['room_type']:20s} -> n={stat_row['n']:6,}, mean={stat_row['mean']:8.3f}, median={stat_row['median']:8.3f}, std={stat_row['std']:8.3f}")

	print("-" * 70)
	print(f"Levene test p-value: {format_p_value(levene_p)}")
	print(f"Selected test: {test_name}")
	print(f"F-statistic: {f_stat:.4f}")
	print(f"p-value: {format_p_value(p_value)}")
	print(f"Eta-squared: {eta_sq:.4f}")

	print("-" * 70)
	print_decision(
		p_value,
		alpha,
		reject_msg="Room type significantly affects price.",
		fail_msg="No significant room type effect detected.",
	)


def test_distance_price_correlation(
	df: pd.DataFrame,
	alpha: float = 0.05,
	distance_col: str = "dist",
) -> None:
	if distance_col not in df.columns:
		print(f"Warning: '{distance_col}' column not found in dataset.")
		return
	if "price" not in df.columns:
		print("Warning: 'price' column not found in dataset.")
		return

	df_clean = df[[distance_col, "price"]].dropna()
	if len(df_clean) < 2:
		print("Warning: Not enough data for correlation test.")
		return

	r_value, p_value = stats.pearsonr(df_clean[distance_col], df_clean["price"])
	r_sq = r_value ** 2

	print("=" * 70)
	print("RQ4: Relationship Between Distance and Price")
	print("H0: rho = 0")
	print("H1: rho != 0")
	print("=" * 70)
	print(f"Distance column: {distance_col}")
	print(f"n: {len(df_clean):,}")
	print(f"Pearson r: {r_value:.4f}")
	print(f"R-squared: {r_sq:.4f}")
	print(f"p-value: {format_p_value(p_value)}")
	print("-" * 70)
	print_decision(
		p_value,
		alpha,
		reject_msg="There is a significant linear relationship.",
		fail_msg="No significant linear relationship detected.",
	)


def test_metro_distance_price_correlation(df: pd.DataFrame, alpha: float = 0.05) -> None:
	test_distance_price_correlation(
		df=df,
		alpha=alpha,
		distance_col="metro_dist",
	)


def test_metro_distance_city_interaction(
	df: pd.DataFrame,
	alpha: float = 0.05,
	metro_col: str = "metro_dist",
	plot_path: Optional[str] = "rq5_metro_city_slopes.png",
) -> None:
	if metro_col not in df.columns:
		print(f"Warning: '{metro_col}' column not found in dataset.")
		return
	if "price" not in df.columns:
		print("Warning: 'price' column not found in dataset.")
		return
	if "city" not in df.columns:
		print("Warning: 'city' column not found in dataset.")
		return

	df_clean = df[["city", metro_col, "price"]].dropna().copy()
	if len(df_clean) < 2:
		print("Warning: Not enough data for interaction model.")
		return

	df_clean["log_price"] = np.log(df_clean["price"])

	cities = sorted(pd.Categorical(df_clean["city"]).categories)
	if not cities:
		print("Warning: No city categories found for interaction model.")
		return
	ref_city = "Berlin" if "Berlin" in cities else cities[0]
	city_term = f"C(city, Treatment(reference='{ref_city}'))"

	model_full = smf.ols(
		"log_price ~ " + metro_col + " * " + city_term,
		data=df_clean,
	).fit()
	model_reduced = smf.ols(
		"log_price ~ " + metro_col + " + " + city_term,
		data=df_clean,
	).fit()

	anova_cmp = sm.stats.anova_lm(model_reduced, model_full)
	if len(anova_cmp) >= 2:
		f_stat = anova_cmp["F"].iloc[1]
		p_value = anova_cmp["Pr(>F)"].iloc[1]
	else:
		f_stat = np.nan
		p_value = np.nan

	interaction_label = f"{metro_col}:{city_term}"
	interaction_rows = []
	for term, coef in model_full.params.items():
		if interaction_label in term:
			interaction_rows.append({
				"term": term,
				"coef": coef,
				"p_value": model_full.pvalues.get(term, np.nan),
			})
	interaction_rows = sorted(interaction_rows, key=lambda r: r["p_value"])
	interaction_sig = sum(1 for row in interaction_rows if row["p_value"] < alpha)

	metro_coef = model_full.params.get(metro_col, np.nan)
	metro_p = model_full.pvalues.get(metro_col, np.nan)
	city_main_rows = []
	for term, coef in model_full.params.items():
		if term.startswith(city_term) and ":" not in term:
			city_main_rows.append({
				"term": term,
				"coef": coef,
				"p_value": model_full.pvalues.get(term, np.nan),
			})
	city_main_sig = sum(1 for row in city_main_rows if row["p_value"] < alpha)

	print_header(
		"RQ5: Metro Distance x City Interaction (log price)",
		[
			"H0: Interaction terms are jointly zero",
			"H1: At least one interaction term is non-zero",
		],
	)
	print(f"Model: log_price ~ {metro_col} + {city_term} + {metro_col}:{city_term}")
	print(f"Reference city: {ref_city}")
	print(f"Metro distance column: {metro_col}")
	print(f"n: {len(df_clean):,}")
	print(f"Model fit: R2={model_full.rsquared:.4f}, Adj R2={model_full.rsquared_adj:.4f}")
	print(
		"Base metro_dist effect (reference city): "
		f"coef={metro_coef:>8.4f}  p={format_p_value(metro_p)}"
	)
	if city_main_rows:
		print(f"City main effects significant at alpha={alpha}: {city_main_sig}")
	print(f"Joint F-test: {f_stat:.4f}")
	print(f"p-value: {format_p_value(p_value)}")
	if interaction_rows:
		print(f"Interaction terms (metro_dist x city): {len(interaction_rows)}")
		print(f"Significant at alpha={alpha}: {interaction_sig}")
	print("-" * 70)
	if interaction_rows:
		print("Top interaction effects (coef, p-value)")
		for row in interaction_rows[:10]:
			print(f"{row['term']:35s} coef={row['coef']:>8.4f}  p={format_p_value(row['p_value'])}")
		print("-" * 70)
	print_decision(
		p_value,
		alpha,
		reject_msg="Metro distance effect varies across cities.",
		fail_msg="No evidence that metro distance effect differs by city.",
	)

	if plot_path:
		plot_metro_city_slopes(
			model_full=model_full,
			df_clean=df_clean,
			metro_col=metro_col,
			output_path=plot_path,
			ref_city=ref_city,
		)
		print(f"Saved plot: {plot_path}")


def plot_metro_city_slopes(
	model_full: sm.regression.linear_model.RegressionResultsWrapper,
	df_clean: pd.DataFrame,
	metro_col: str,
	output_path: str,
	ref_city: str,
) -> None:
	if "city" not in df_clean.columns:
		return

	cities = sorted(pd.Categorical(df_clean["city"]).categories)
	if not cities:
		return

	cov = model_full.cov_params()
	rows = []
	if ref_city not in cities:
		ref_city = cities[0]
	for city in cities:
		if city == ref_city:
			continue
		term = f"{metro_col}:C(city, Treatment(reference='{ref_city}'))[T.{city}]"
		term_coef = model_full.params.get(term, np.nan)
		term_var = cov.loc[term, term] if term in cov.index else np.nan
		se = np.sqrt(term_var) if np.isfinite(term_var) and term_var >= 0 else np.nan
		rows.append({
			"city": city,
			"effect": term_coef,
			"se": se,
		})

	plot_df = pd.DataFrame(rows).sort_values("effect")
	ci_half = 1.96 * plot_df["se"]

	fig_height = max(4.0, 0.45 * len(plot_df))
	fig, ax = plt.subplots(figsize=(10, fig_height))
	positions = np.arange(len(plot_df))

	ax.errorbar(
		plot_df["effect"],
		positions,
		xerr=ci_half,
		fmt="o",
		color="#1f77b4",
		ecolor="#8da0cb",
		elinewidth=2,
		capsize=3,
	)
	ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
	ax.set_yticks(positions)
	ax.set_yticklabels(plot_df["city"])
	ax.set_xlabel("Interaction effect vs reference city")
	ax.set_title("Metro distance x city interaction coefficients (95% CI)")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def run_log_price_regression(
	df: pd.DataFrame,
	alpha: float = 0.05,
	plot_path: str = "regression_residuals.png",
) -> None:
	if "price" not in df.columns:
		print("Warning: 'price' column not found in dataset.")
		return

	terms = []
	if "room_type" in df.columns:
		terms.append("C(room_type)")
	if "dist" in df.columns:
		terms.append("dist")
	if "metro_dist" in df.columns:
		terms.append("metro_dist")
	if "person_capacity" in df.columns:
		terms.append("person_capacity")
	for col in ["host_is_superhost", "multi", "biz"]:
		if col in df.columns:
			terms.append(col)
	ref_city = None
	if "city" in df.columns:
		city_means = df.groupby("city")["price"].mean().sort_values()
		if len(city_means) > 0:
			median_price = city_means.median()
			ref_city = (city_means - median_price).abs().idxmin()
			terms.append(f"C(city, Treatment(reference='{ref_city}'))")

	if not terms:
		print("Warning: No explanatory variables found for regression.")
		return

	model_cols = ["price"] + [c for c in df.columns if c in [
		"room_type",
		"dist",
		"metro_dist",
		"person_capacity",
		"host_is_superhost",
		"multi",
		"biz",
		"city",
	]]
	model_df = df[model_cols].dropna().copy()
	if len(model_df) < 2:
		print("Warning: Not enough data for regression model.")
		return

	model_df["log_price"] = np.log(model_df["price"])
	formula = "log_price ~ " + " + ".join(terms)
	model = smf.ols(formula, data=model_df).fit()

	print_header("RQ6: Multiple Regression on Log(Price)", [f"Model: {formula}"])
	if ref_city:
		print(f"Reference city: {ref_city} (median price)")
	print("-" * 70)
	print(f"n: {len(model_df):,}")
	print(f"R-squared: {model.rsquared:.4f}")
	print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
	print("-" * 70)
	print("Approx. effects (log-price coefficients with 95% CI)")
	coef_df = model.params.to_frame(name="coef")
	coef_df["lower"] = model.conf_int()[0]
	coef_df["upper"] = model.conf_int()[1]
	coef_df = coef_df.drop(index=["Intercept"], errors="ignore")
	coef_df = coef_df.rename(index={
		"C(room_type)[T.Private room]": "Room type: Private room",
		"C(room_type)[T.Shared room]": "Room type: Shared room",
	})
	if ref_city:
		coef_df = coef_df.rename(index={
			f"C(city, Treatment(reference='{ref_city}'))[T.{ref_city}]": f"City: {ref_city} (ref)",
		})
		if f"City: {ref_city} (ref)" not in coef_df.index:
			coef_df.loc[f"City: {ref_city} (ref)"] = {
				"coef": 0.0,
				"lower": 0.0,
				"upper": 0.0,
			}
	if "C(room_type)[T.Entire home/apt]" not in coef_df.index:
		coef_df.loc["Room type: Entire home/apt (ref)"] = {
			"coef": 0.0,
			"lower": 0.0,
			"upper": 0.0,
		}
	coef_df = coef_df.sort_values("coef")
	for name, row in coef_df.iterrows():
		pct = (np.exp(row["coef"]) - 1) * 100
		pct_lo = (np.exp(row["lower"]) - 1) * 100
		pct_hi = (np.exp(row["upper"]) - 1) * 100
		print(
			f"{name:35s} coef={row['coef']:>8.3f}  CI=[{row['lower']:>8.3f}, {row['upper']:>8.3f}]  "
			f"~{pct:>7.1f}% (CI {pct_lo:>6.1f}% to {pct_hi:>6.1f}%)"
		)
	print("-" * 70)

	plt.figure(figsize=(7, 5))
	plt.scatter(model.fittedvalues, model.resid, alpha=0.3, edgecolors="none")
	plt.axhline(0.0, color="red", linestyle="--", linewidth=1)
	plt.xlabel("Fitted values (log price)")
	plt.ylabel("Residuals")
	plt.title("Residuals vs Fitted (Log Price Regression)")
	plt.tight_layout()
	plt.savefig(plot_path, dpi=150)
	plt.close()
	print(f"Saved plot: {plot_path}")

	coef_plot_path = "regression_coefficients.png"
	plt.figure(figsize=(8, max(4, 0.25 * len(coef_df))))
	y_pos = np.arange(len(coef_df))
	plt.errorbar(
		coef_df["coef"].values,
		y_pos,
		xerr=[
			coef_df["coef"].values - coef_df["lower"].values,
			coef_df["upper"].values - coef_df["coef"].values,
		],
		fmt="o",
		color="#1f77b4",
		elinewidth=1,
		capsize=2,
	)
	plt.axvline(0.0, color="red", linestyle="--", linewidth=1)
	plt.yticks(y_pos, coef_df.index)
	plt.xlabel("Coefficient (log price)")
	plt.title("Coefficient Plot (95% CI)")
	plt.tight_layout()
	plt.savefig(coef_plot_path, dpi=150)
	plt.close()
	print(f"Saved plot: {coef_plot_path}")
	print("-" * 70)
	print_decision(
		model.f_pvalue,
		alpha,
		reject_msg="The predictors jointly explain log(price).",
		fail_msg="No evidence that the predictors jointly explain log(price).",
	)


if __name__ == "__main__":
	data_folder = Path("Yeni klasor")
	if not data_folder.exists():
		# Keep compatibility with existing folder name that may include Unicode char.
		data_folder = Path("Yeni klasör")

	merged_df = load_merged_data(data_folder)
	print(f"Loaded rows: {len(merged_df):,}")
	print(f"Cities: {sorted(merged_df['city'].unique().tolist())}")
	print(f"Day types: {sorted(merged_df['day_type'].unique().tolist())}")
	print(f"Room types: {sorted(merged_df['room_type'].unique().tolist())}")
	print()

	test_weekday_vs_weekend_price(merged_df)
	print("\n")
	test_room_type_anova(merged_df)
	print("\n")
	test_city_anova_price(merged_df)
	print("\n")
	test_city_anova_satisfaction(merged_df)
	print("\n")
	test_distance_price_correlation(merged_df)
	print("\n")
	test_metro_distance_price_correlation(merged_df)
	print("\n")
	test_metro_distance_city_interaction(merged_df)
	print("\n")
	run_log_price_regression(merged_df)