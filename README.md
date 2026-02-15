# FraudGuard — Credit Card Fraud Detection (Streamlit)

FraudGuard is an interactive Streamlit web app that detects suspicious credit card transactions using a trained XGBoost model. It provides a clean dashboard for exploring predictions, viewing model performance, and reviewing flagged transactions.

The app loads the dataset dynamically from Kaggle via the Croissant metadata interface (no need to ship the large CSV in the repo).

## Demo

- **Streamlit Community Cloud**: *([app link](https://cc-fraud-rohan-nagaonkar.streamlit.app/))*

## Features

- **Dynamic dataset loading** from Kaggle (Croissant)
- **On-the-fly model training** with XGBoost
- **Imbalance handling** using under-sampling + SMOTE
- **Interactive visualizations** with Plotly
- **Fraud list view** to inspect flagged transactions

## Tech Stack

- **Frontend/App**: Streamlit
- **Data**: Pandas, NumPy
- **Modeling**: scikit-learn, XGBoost, imbalanced-learn
- **Visualization**: Plotly
- **Data source**: Kaggle + `mlcroissant`

## Project Structure

```text
.
├── app.py
├── requirements.txt
└── README.md
```

## Setup (Local)

### 1) Clone the repository

```bash
git clone https://github.com/rohan2004nagaonkar/cc-fraud.git
cd cc-fraud
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the app

```bash
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Push your code to GitHub (this repo).
2. Go to https://share.streamlit.io
3. Click **New app**
4. Select:
   - **Repository**: `rohan2004nagaonkar/cc-fraud`
   - **Branch**: `main`
   - **Main file path**: `app.py`

Streamlit will install packages from `requirements.txt` automatically.

## Notes on the Dataset

- The app is configured to load the dataset using the Kaggle Croissant URL in `app.py` (`KAGGLE_CROISSANT_URL`).
- If the Croissant loader returns prefixed column names (e.g., `creditcard.csv/Amount`), the app normalizes them back to `Amount`, `Time`, and `Class`.

## Screenshots

Add images to a folder (e.g., `assets/`) and reference them here.

```md
![Dashboard](assets/dashboard.png)
```

## License

Add a license if you plan to make this public.
