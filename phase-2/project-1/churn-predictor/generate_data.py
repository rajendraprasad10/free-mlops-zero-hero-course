"""Generate a realistic synthetic telecom customer churn dataset -> data/churn.csv"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 5000

gender = rng.choice(["Male", "Female"], N)
senior = rng.choice([0, 1], N, p=[0.84, 0.16])
partner = rng.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = rng.choice(["Yes", "No"], N, p=[0.30, 0.70])
tenure = np.clip(rng.gamma(2.2, 14, N).astype(int), 1, 72)
contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
internet = rng.choice(["Fiber optic", "DSL", "No"], N, p=[0.44, 0.34, 0.22])
phone = rng.choice(["Yes", "No"], N, p=[0.90, 0.10])
tech_support = np.where(internet == "No", "No internet",
                        rng.choice(["Yes", "No"], N, p=[0.35, 0.65]))
online_security = np.where(internet == "No", "No internet",
                           rng.choice(["Yes", "No"], N, p=[0.35, 0.65]))
paperless = rng.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    N, p=[0.34, 0.23, 0.22, 0.21])

base = np.select([internet == "Fiber optic", internet == "DSL"], [70, 45], default=20)
monthly = np.round(base + (phone == "Yes") * 8 + rng.normal(0, 6, N), 2).clip(18, 120)
total = np.round(monthly * tenure + rng.normal(0, 25, N), 2).clip(18, None)

# churn probability via a logistic model with real-world-ish drivers
z = (-1.2
     + 1.3 * (contract == "Month-to-month")
     - 1.1 * (contract == "Two year")
     + 0.55 * (internet == "Fiber optic")
     - 0.035 * tenure
     + 0.012 * (monthly - 65)
     + 0.35 * (payment == "Electronic check")
     + 0.30 * (tech_support == "No")
     + 0.25 * (online_security == "No")
     + 0.20 * senior
     + 0.10 * (paperless == "Yes")
     - 0.20 * (dependents == "Yes")
     + rng.normal(0, 0.5, N))
churn = rng.random(N) < 1 / (1 + np.exp(-z))

df = pd.DataFrame({
    "customerID": [f"CUST-{i:05d}" for i in range(1, N + 1)],
    "gender": gender, "SeniorCitizen": senior, "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
    "InternetService": internet, "OnlineSecurity": online_security,
    "TechSupport": tech_support, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total,
    "Churn": np.where(churn, "Yes", "No"),
})

out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
df.to_csv(out / "churn.csv", index=False)
print(f"Wrote {len(df)} rows -> {out / 'churn.csv'} | churn rate: {churn.mean():.1%}")
