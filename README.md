# EmberWatch - Fire & Smoke Detection

Streamlit dashboard for the Logistic Regression model trained in `Fire_Smoke_Detection.ipynb`
(StandardScaler + LogisticRegression, ~98.8% test accuracy).

## Features
- Live prediction with sliders and exact inputs, a probability gauge and risk levels (Safe, Watch, Warning, Critical)
- Adjustable alert sensitivity (decision threshold)
- Explanation chart showing which sensor pushed the result towards or away from fire
- What-if analysis with the exact trigger point for any reading
- Batch CSV prediction with column matching, validation and downloadable results
- Model insights: metrics, confusion matrix, ROC curve, feature weights, data balance
- Session log with download
- Input validation: impossible values are rejected, values outside the training range raise a warning

## Project structure
```
app.py                 main app
requirements.txt       dependencies (scikit-learn is pinned to the training version)
Fire_Detection.pki     trained model (app also accepts Fire_Detection.pkl)
fire_smoke_detection_logistic_regression_dataset.csv   used for the Model insights tab
.streamlit/config.toml theme and server settings
Dockerfile             container deployment
```

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

### 1. Streamlit Community Cloud (easiest, free)
1. Push this folder to a GitHub repo (keep `app.py` and `requirements.txt` in the root).
2. Go to share.streamlit.io, click **Create app**, pick the repo, branch and `app.py`.
3. In **Advanced settings** choose Python 3.11 or 3.12, then deploy.

### 2. Render (Docker)
1. Push to GitHub, create a **New Web Service**, choose **Docker** as runtime.
2. Render sets `$PORT` automatically; the Dockerfile already reads it.

### 3. Hugging Face Spaces (Docker)
1. Create a Space with the **Docker** SDK and upload these files.
2. Add this header at the top of README.md: `app_port: 8501` inside a YAML block (`---` lines).

### 4. Any server with Docker
```bash
docker build -t emberwatch .
docker run -p 8501:8501 emberwatch
```

## Notes
- The model was trained with **scikit-learn 1.6.1**. Keep that pin, otherwise loading may fail or warn.
- Only load pickle/joblib files you trust.
- The dataset has Smoke_PPM, CO_PPM and Flame_Intensity, which this model does not use. Retraining with them can improve detection.
- Decision-support demo only. It does not replace certified smoke alarms.
