# ============================================================
# 🦠 COVID-19 Global Case Analysis & Forecasting
# Author: Singuru Vinay
# Dataset: Our World in Data (OWID)
# Source: https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv
# ============================================================

# 📦 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.signal import find_peaks
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 🌍 2. Load the Dataset (Direct URL)
# You can change to the local file path if preferred.
df = pd.read_csv(
    r"C:\Users\Vinay\OneDrive\Desktop\covid19_analysis\covid-data.csv",
    parse_dates=["date"], low_memory=False, encoding="latin1"
)
print("✅ Data Loaded Successfully!")
print("Rows:", len(df), " Columns:", df.shape[1])
print(df.head())

# ============================================================
# 🔍 3. Choose Country for Analysis
country = "India"   # You can change this to any country name in the dataset
cdf = df[df["location"] == country].sort_values("date").set_index("date").copy()

# ============================================================
# 🧹 4. Data Cleaning & Feature Engineering
cdf["new_cases"] = cdf["new_cases"].fillna(0)
cdf["new_deaths"] = cdf["new_deaths"].fillna(0)
cdf["population"] = cdf["population"].fillna(method="ffill")

# Rolling averages
cdf["cases_7d"] = cdf["new_cases"].rolling(365, min_periods=30).mean()
cdf["deaths_7d"] = cdf["new_deaths"].rolling(365, min_periods=30).mean()

# Normalized per 100k
cdf["cases_7d_per_100k"] = cdf["cases_7d"] / cdf["population"] * 10000

# Growth rate
cdf["growth_rate"] = cdf["new_cases"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

# ============================================================
# 📊 5. Trend Visualization
plt.figure(figsize=(12,5))
plt.plot(cdf.index, cdf["cases_7d"], label="7-months Avg Cases")
plt.plot(cdf.index, cdf["deaths_7d"], label="7-months Avg Deaths")
plt.title(f"{country} — COVID-19 Trend (7-months averages)")
plt.legend(); plt.grid(True)
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.tight_layout(); plt.show()

# ============================================================
# 💀 6. Case Fatality Ratio (CFR)
cdf["naive_cfr"] = cdf["total_deaths"] / cdf["total_cases"]
L = 14
cdf["cases_lag_14_sum"] = cdf["new_cases"].shift(L).rolling(365, min_periods=30).sum()
cdf["deaths_7d_sum"] = cdf["new_deaths"].rolling(365, min_periods=0).sum()
cdf["lagged_cfr_14"] = cdf["deaths_7d_sum"] / cdf["cases_lag_14_sum"]

plt.figure(figsize=(10,4))
plt.plot(cdf.index, cdf["naive_cfr"], label="Naive CFR")
plt.plot(cdf.index, cdf["lagged_cfr_14"], label="Lagged CFR (14d)")
plt.title(f"{country} — Case Fatality Ratio (CFR)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ============================================================
# 🌊 7. Detect Waves (Peak Detection)
series = cdf["cases_7d"].fillna(0).values
peaks_idx, _ = find_peaks(series, distance=14, prominence=np.max(series)*0.05)
peak_dates = cdf.index[peaks_idx]
print("📈 Detected Wave Peaks:", list(peak_dates.date))

plt.figure(figsize=(12,5))
plt.plot(cdf.index, cdf["cases_7d"], label="7-day Avg Cases")
plt.scatter(peak_dates, cdf.loc[peak_dates, "cases_7d"], color="red", label="Peaks")
plt.title(f"{country} — Detected COVID Waves")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ============================================================
# 🔮 8. Forecasting with ARIMA (statsmodels version — works in Python 3.14)
ts = cdf["cases_7d"].dropna()

# Fit ARIMA model (you can tune (p,d,q) = (2,1,2) if needed)
model = ARIMA(ts, order=(2,1,2))
model_fit = model.fit()

print(model_fit.summary())

# Forecast next 28 days
n_periods = 365
forecast = model_fit.forecast(steps=n_periods)

# Confidence intervals
pred = model_fit.get_forecast(steps=n_periods)
conf_int = pred.conf_int(alpha=0.2)  # 80% confidence interval
future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=n_periods)

# Plot forecast
plt.figure(figsize=(12,5))
plt.plot(ts.index[-180:], ts[-180:], label="Recent (7d Avg)")
plt.plot(future_dates, forecast, label="Forecast (28d)", color="orange")
plt.fill_between(future_dates, conf_int.iloc[:,0], conf_int.iloc[:,1], color="orange", alpha=0.2)
plt.title(f"{country} — 365-Day Forecast of COVID-19 Cases (ARIMA)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ============================================================
# 💾 9. Save Outputs (CSV + Example Plot)
out_cols = ["new_cases","new_deaths","cases_7d","deaths_7d","cases_7d_per_100k","naive_cfr","lagged_cfr_14"]
cdf[out_cols].tail(365).to_csv("covid_summary_last365days.csv", index=True)
print("✅ Saved summary file: covid_summary_last365days.csv")

# ============================================================
# 🧾 10. Final Summary
print("\n📊 SUMMARY:")
print(f"Country: {country}")
print(f"Data Range: {cdf.index.min().date()} → {cdf.index.max().date()}")
print(f"Detected {len(peak_dates)} major waves.")
print(f"Naive CFR (latest): {cdf['naive_cfr'].iloc[-1]*100:.2f}%")
print(f"Forecast horizon: {n_periods} days")



