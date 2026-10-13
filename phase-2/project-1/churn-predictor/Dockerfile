FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the dataset and model at image build time if they are missing
RUN [ -f data/churn.csv ] || python generate_data.py; \
    [ -f model/model.joblib ] || python train.py

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health')"

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
