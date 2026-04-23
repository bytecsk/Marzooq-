# 🔍 Fraud Detection — Streamlit App

An interactive machine-learning web application that detects fraudulent transactions using a **Random Forest** or **Logistic Regression** classifier, built with **Streamlit** and **scikit-learn**.

---

## 📂 Project Structure

```
fraud-detection-app/
├── app.py                        # Main Streamlit application
├── fraud_detection_cleaned.csv   # Dataset (place here before running)
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧩 Variables

### Dependent Variable (Target)
| Variable | Description |
|---|---|
| `Fraud_Label` | **1** = Fraudulent transaction, **0** = Legitimate transaction |

### Independent Variables (Features)
| Variable | Description |
|---|---|
| `Transaction_Amount` | Amount of the transaction |
| `Transaction_Type` | Encoded transaction type (0–4) |
| `Account_Balance` | Account balance at transaction time |
| `Device_Type` | Device used (encoded) |
| `Location` | Transaction location (encoded) |
| `Merchant_Category` | Merchant category code |
| `IP_Address_Flag` | 1 = suspicious IP address |
| `Previous_Fraudulent_Activity` | Prior fraud count for the account |
| `Daily_Transaction_Count` | Number of transactions today |
| `Avg_Transaction_Amount_7d` | 7-day rolling average transaction amount |
| `Failed_Transaction_Count_7d` | Failed transactions in the last 7 days |
| `Card_Type` | Card type (encoded) |
| `Card_Age` | Age of card in months |
| `Transaction_Distance` | Distance from home location (km) |
| `Authentication_Method` | Authentication method used (encoded) |
| `Risk_Score` | Pre-computed risk score (0–1) |
| `Is_Weekend` | 1 = weekend transaction |
| `Hour` | Hour of the day (0–23) |
| `DayOfWeek` | Day of week (0 = Monday) |
| `Month` | Month of year (1–12) |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fraud-detection-app.git
cd fraud-detection-app
```

### 2. Add the dataset
Place `fraud_detection_cleaned.csv` in the project root directory.

### 3. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🖥️ App Pages

| Page | Description |
|---|---|
| **📊 Data Overview** | Dataset stats, variable glossary, correlation heatmap, feature distributions |
| **🤖 Train Model** | Train Random Forest or Logistic Regression; view feature importances |
| **📈 Evaluation** | Confusion matrix, ROC curve, classification report, score distributions |
| **🔮 Predict** | Enter transaction details to get a real-time fraud prediction |

---

## ⚙️ Model Settings (Sidebar)
- **Algorithm** — Random Forest or Logistic Regression
- **Test Size** — Fraction of data held out for evaluation (10–40%)
- **Random State** — Seed for reproducibility

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/)
- [joblib](https://joblib.readthedocs.io/)

---

## 📊 Dataset
- **Records:** 50,000 transactions
- **Fraud rate:** ~32%
- **Features:** 20 independent variables
- **Target:** `Fraud_Label` (binary)

---

## 📄 License
MIT License — free to use and modify.
