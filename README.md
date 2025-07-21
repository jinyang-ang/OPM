# [OPM Tools Suite](https://ajy-opm.streamlit.app/), 

A collection of Python-based financial analysis tools, including:

* **OPM** – A Sell‑Only Option Analysis dashboard
* **ACIC** – Annual Compound Interest Calculator
* **Box Method** – Trade‑Probability Box Method Explorer

*All efforts by jinyang-ang.*

---

## 📦 Repository Structure

```
OPM/
├── .devcontainer/         # VS Code dev‑container settings
├── pages/                 # Streamlit pages for ACIC & Box Method
│   ├── ACIC.py
│   └── Box Method.py
├── OPM.py                 # Main Streamlit app for OPM
├── requirements.txt       # Python dependencies
└── README.md              # (This file)
```

---

## 🚀 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/jinyang-ang/OPM.git
   cd OPM
   ```

2. **Create & activate a Python 3.8+ virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   # or
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠 Tools Overview

### 1. OPM — Sell‑Only Option Analysis

A Streamlit dashboard to analyze selling option premiums:

* Fetches historical price data via `yfinance`
* Supports line/candlestick charts with current‑price overlay
* Displays next earnings date and tags expiries in the earnings week
* Computes call/put market premiums, breakevens, and highlights ATM

**Run**

```bash
streamlit run OPM.py
```

---

### 2. ACIC — Annual Compound Interest Calculator

Located in `pages/ACIC.py`, ACIC is designed to:

* Input initial principal, annual interest rate, compounding frequency, and investment duration
* Calculate compound growth and final account value
* Generate an amortization schedule showing balance and interest breakdown per period
* Export results to CSV for further analysis

**Run**

```bash
streamlit run pages/ACIC.py
```

---

### 3. Box Method — Trade‑Probability Explorer

Located in `pages/Box Method.py`, the Box Method tool:

* Builds a probability “box” around recent price moves
* Calculates event frequencies inside/outside the box
* Visualizes hit‑rate vs. expiration dates
* Allows adjustable lookback window and probability thresholds

**Run**

```bash
streamlit run pages/Box Method.py
```

---
