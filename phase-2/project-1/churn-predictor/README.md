# 📉 Customer Churn Predictor

An end-to-end machine learning project: a CSV dataset, a trained scikit-learn model,
and a polished Streamlit web app that is ready to deploy.

## Features
- **Predict**: enter one customer's profile, get a churn probability gauge, risk band and retention tips
- **Batch scoring**: upload a CSV, score every customer, download the results
- **Data explorer**: interactive churn-rate and distribution charts over `data/churn.csv`
- **Model performance**: accuracy, precision, recall, F1, ROC-AUC, ROC curve, confusion matrix, feature importance

## Project structure
```
churn-predictor/
├── app.py               # Streamlit UI
├── generate_data.py     # builds data/churn.csv (5,000 synthetic customers)
├── train.py             # trains the model -> model/model.joblib + metrics.json
├── data/churn.csv       # dataset
├── model/               # trained pipeline + metrics
├── requirements.txt
├── Dockerfile
├── Procfile             # Heroku / Render style platforms
└── .streamlit/config.toml
```

## Run locally
```bash
pip install -r requirements.txt
python generate_data.py   # optional, CSV already included
python train.py           # optional, model already included
streamlit run app.py
```
Open http://localhost:8501.

## Deploy

### Streamlit Community Cloud (free, easiest)
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, click **New app**, select the repo and `app.py`.
3. Deploy. The model and CSV are committed, so no extra setup is needed.

### Docker (any cloud: Cloud Run, ECS, Azure, Fly.io, a VPS)
```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```
The container reads the `PORT` environment variable, so it works on Cloud Run and similar platforms.

### Render / Railway / Heroku
Use the included `Procfile` (start command) and `requirements.txt` (build).

## Using your own data
Replace `data/churn.csv` with your file (keep the same columns, or edit `NUMERIC` /
`CATEGORICAL` in `train.py` and `app.py`), then run `python train.py`.

## Notes
The dataset is synthetic and generated for demonstration. The model is a Gradient Boosting
pipeline (scaling + one-hot encoding + classifier) saved as a single joblib file.
