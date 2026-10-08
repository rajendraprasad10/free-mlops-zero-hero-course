# House Price Prediction API

User → FastAPI → ML model (`model.pkl`) → Prediction

## python setup in aws server 

sudo apt update
sudo apt install -y python3 python3-pip python3-venv 


## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py              # optional: regenerates data/houses.csv
python train.py                      # reads the CSV, creates model/model.pkl
uvicorn app.main:app --reload        # UI at http://localhost:8000, docs at /docs
```

## Run with Docker
```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
# or: docker compose up --build
```

## Call the API
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"area_sqft":1800,"bedrooms":3,"bathrooms":2,"age_years":10,"location_score":7,"garage":1}'
```

## What to learn here
- **Serialization**: `joblib.dump(pipeline, "model/model.pkl")` saves the scaler *and* model together, so inference gets identical preprocessing.
- **Serving**: load the model once in FastAPI's `lifespan`, validate input with Pydantic, return typed responses.
- **Docker**: training runs at build time so `model.pkl` matches the installed scikit-learn version. Never unpickle files from untrusted sources.
- **Next steps**: use a real dataset (e.g. Ames or Kaggle House Prices), add tests, version models, add logging/monitoring.

## Using your own CSV
Put a file anywhere with these columns (header row required):
`area_sqft, bedrooms, bathrooms, age_years, location_score, garage, price`

```bash
python train.py --data path/to/your.csv
```
`train.py` checks the columns, drops duplicates and rows without a price, and fills missing feature values
with the median (the imputer is saved inside `model.pkl`). For Docker, replace `data/houses.csv` and rebuild.
If your real dataset has other columns, edit `FEATURES` in `train.py` and the `House` schema in `app/main.py`.
