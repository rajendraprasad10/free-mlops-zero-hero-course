"""Create data/houses.csv. Run once, or replace the CSV with your own real data."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5000
area = rng.normal(1800, 600, n).clip(400, 6000).round()
bedrooms = np.clip((area / 600 + rng.normal(0, 0.7, n)).round(), 1, 6).astype(int)
bathrooms = np.clip((bedrooms * 0.7 + rng.normal(0, 0.5, n)).round(), 1, 5).astype(int)
age = rng.uniform(0, 60, n).round()
location = rng.integers(1, 11, n)
garage = rng.integers(0, 4, n)
price = (area * 110 + bedrooms * 8000 + bathrooms * 12000 - age * 1500
         + location * 28000 + garage * 9000 + location * area * 6
         + rng.normal(0, 15000, n)).clip(30000).round()

df = pd.DataFrame({"area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                   "age_years": age, "location_score": location, "garage": garage, "price": price})

# Make it realistic: a few missing values and duplicate rows to clean later
df.loc[rng.choice(n, 50, replace=False), "age_years"] = np.nan
df.loc[rng.choice(n, 30, replace=False), "garage"] = np.nan
df = pd.concat([df, df.sample(20, random_state=1)], ignore_index=True)

df.to_csv("data/houses.csv", index=False)
print(f"Wrote data/houses.csv with {len(df)} rows")
