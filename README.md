# 🦠 COVID-19 Global Case Analysis & Forecasting

A comprehensive **COVID-19 global data analysis and forecasting** project built using **Python**.  
It analyzes trends, waves, and case fatality ratios, and predicts future cases using an **ARIMA model**.

---

## 📖 Project Overview

This project processes real-time COVID-19 data from **Our World in Data (OWID)** and performs:
- Trend analysis (mothly, weekly averages)
- Case Fatality Ratio (CFR) estimation
- Detection of major COVID-19 waves
- 365-day ARIMA forecasting
- Automated plots and data summaries

The project is designed to be:
✅ **Fully ready-to-run**  
✅ **GitHub-friendly**  
✅ **Colab-compatible**

---

## 🌍 Dataset Information

**Source:** [Our World in Data (OWID) COVID-19 Dataset](https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv)

Direct CSV Link (used in the code):
https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv

---

## ⚙️ Installation & Setup

### 🔸 Option 1: Google Colab (Recommended)
1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy & paste the full script from `covid19_case_analysis.py`
4. Run all cells

### 🔸 Option 2: Local (Jupyter / VS Code)
```bash
git clone https://github.com/<your-username>/covid19-global-analysis.git
cd covid19-global-analysis
pip install pandas numpy matplotlib scipy pmdarima
python covid19_case_analysis.py
```

---

## 📊 Output & Visualizations

1️⃣ Trend Analysis (7monthsy rolling average of new cases)  
2️⃣ Case Fatality Ratio (CFR) analysis  
3️⃣ Wave detection using peaks  
4️⃣ 368-day forecasting using ARIMA

---

## 📂 Output Files

| File | Description |
|------|--------------|
| `covid_summary_last365days.csv` | Summary of recent COVID-19 data |

---

## 📅 Example Country
```python
country = "India"  # Change to "United States", "Brazil", "Germany", etc.
```

---

## 🧠 Model Used

- **ARIMA (AutoRegressive Integrated Moving Average)** from `pmdarima`
- Automatically optimizes model parameters for forecasting

---

## 🧑‍💻 Author

**Name:** Singuru Vinay  
**LinkedIn:** [linkedin.com/in/singuru-vinay-57050125b](https://linkedin.com/in/singuru-vinay-57050125b)  
**License:** MIT License  

---

## 🏁 Acknowledgements

- Dataset: Our World in Data (OWID)  
- Libraries: Pandas, NumPy, Matplotlib, SciPy, statsmodels.tsa.arima.model  
- Developed by: *Singuru Vinay*  

---

**🎯 “Data is the new microscope — it helps us see the invisible patterns of the world.”**
