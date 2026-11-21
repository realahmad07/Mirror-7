"""Parameter-Free Tabular, CSV & Time-Series Statistical Data Analyst.

Performs deterministic statistical analysis, MAD/IQR outlier detection, exact OLS
linear trend regression (y = a*x + b with R^2), growth-rate calculation, and next-step
forecasting in English and Roman Urdu with 0 neural parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import statistics
from typing import Any


@dataclass(frozen=True)
class DataAnalysisReport:
    matched: bool
    count: int = 0
    total_sum: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    growth_pct: float = 0.0
    outliers: tuple[float, ...] = ()
    slope: float = 0.0
    intercept: float = 0.0
    r_squared: float = 0.0
    next_forecast: float = 0.0
    primary_output: float | None = None
    explanation: str = ""
    uses_neural_parameters: bool = False


def _detect_outliers(values: list[float]) -> tuple[float, ...]:
    if len(values) < 4:
        return ()
    med = statistics.median(values)
    abs_devs = [abs(v - med) for v in values]
    mad = statistics.median(abs_devs)
    outliers: list[float] = []
    if mad > 0:
        for v in values:
            mod_z = 0.6745 * abs(v - med) / mad
            if mod_z > 3.5:
                outliers.append(v)
    else:
        # Fallback to IQR if MAD == 0
        s = sorted(values)
        q1 = s[len(s) // 4]
        q3 = s[(3 * len(s)) // 4]
        iqr = q3 - q1
        for v in values:
            if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr:
                outliers.append(v)
    return tuple(outliers)


def _ols_linear_trend(values: list[float]) -> tuple[float, float, float, float]:
    """Fit y = a*x + b for x = 1..n. Returns (slope_a, intercept_b, r_squared, next_forecast)."""
    n = len(values)
    if n < 2:
        v = values[0] if values else 0.0
        return 0.0, v, 1.0, v
    xs = [float(i + 1) for i in range(n)]
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_yy = sum((y - y_mean) ** 2 for y in values)
    slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
    intercept = y_mean - slope * x_mean
    if ss_yy == 0:
        r2 = 1.0
    else:
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, values))
        r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_yy)))
    next_val = slope * (n + 1) + intercept
    return round(slope, 6), round(intercept, 6), round(r2, 4), round(next_val, 6)


class ZeroParamDataAnalyst:
    """Zero-parameter statistical, outlier, and OLS linear trend analyzer."""

    VERSION = "zero-param-data-analyst-v1"

    def analyze_prompt(self, prompt: str) -> DataAnalysisReport:
        text = prompt.strip()
        lower = text.casefold()

        # Trigger guard: only run on explicit data analysis / series / outlier prompts
        is_data_query = bool(
            re.match(
                r"^\s*(?:analyze\s+(?:sales\s+|financial\s+|csv\s+|tabular\s+)?data|data\s+ka\s+tajziya|"
                r"find\s+outliers?\s+in(?:\s+data)?|predict\s+next(?:\s+value)?(?:\s+for\s+series)?|"
                r"forecast\s+next(?:\s+value)?)",
                lower,
            )
        )
        if not is_data_query:
            return DataAnalysisReport(False)

        # Extract numeric series after ':' or from list/pairs
        payload = text.split(":", 1)[1] if ":" in text else text
        # Extract numbers (handling key=value like Jan=100 or plain lists [10, 20, 30])
        kv_matches = re.findall(r"[A-Za-z]+\s*=\s*(-?\d+(?:\.\d+)?)", payload)
        if len(kv_matches) >= 3:
            values = [float(x) for x in kv_matches]
        else:
            raw_nums = re.findall(r"-?\d+(?:\.\d+)?", payload)
            values = [float(x) for x in raw_nums]

        if len(values) < 3:
            return DataAnalysisReport(False)

        n = len(values)
        total = round(sum(values), 6)
        mean_v = round(total / n, 6)
        med_v = round(float(statistics.median(values)), 6)
        min_v = min(values)
        max_v = max(values)
        growth = round(((values[-1] - values[0]) / abs(values[0])) * 100.0, 2) if values[0] != 0 else 0.0
        outliers = _detect_outliers(values)
        slope, intercept, r2, next_forecast = _ols_linear_trend(values)

        # Choose primary scalar output based on query intent
        if "outlier" in lower:
            primary = outliers[0] if outliers else 0.0
        elif "predict" in lower or "forecast" in lower:
            primary = next_forecast
        elif "median" in lower:
            primary = med_v
        elif "sum" in lower or "total" in lower:
            primary = total
        else:
            primary = mean_v

        is_urdu = bool(re.search(r"\b(?:urdu|tajziya|batao|karo)\b", lower))
        outlier_str = ", ".join(f"{o:g}" for o in outliers) if outliers else "None"

        if is_urdu:
            expl = (
                f"Data Tajziya Report (0-Parameter Statistical & OLS Engine — N={n}):\n"
                f"- Sum: {total:g} | Mean (Ausat): {mean_v:g} | Median: {med_v:g} | Min..Max: [{min_v:g}, {max_v:g}]\n"
                f"- Overall Growth: {growth:g}% | Detected Outliers (MAD/IQR): {outlier_str}\n"
                f"- OLS Linear Trend: y = {slope:g}*t + {intercept:g} (R^2 = {r2:.4f}) -> Next Forecast (t={n+1}): {next_forecast:g}"
            )
        else:
            expl = (
                f"Verified Data Analysis Report (0-Parameter Statistical & OLS Engine — N={n}):\n"
                f"- Sum: {total:g} | Mean: {mean_v:g} | Median: {med_v:g} | Range: [{min_v:g}, {max_v:g}]\n"
                f"- Overall Growth: {growth:g}% | Outliers (MAD/IQR): {outlier_str}\n"
                f"- OLS Linear Trend: y = {slope:g}*t + {intercept:g} (R^2 = {r2:.4f}) -> Next Forecast (t={n+1}): {next_forecast:g}"
            )

        return DataAnalysisReport(
            matched=True,
            count=n,
            total_sum=total,
            mean=mean_v,
            median=med_v,
            min_val=min_v,
            max_val=max_v,
            growth_pct=growth,
            outliers=outliers,
            slope=slope,
            intercept=intercept,
            r_squared=r2,
            next_forecast=next_forecast,
            primary_output=primary,
            explanation=expl,
        )
