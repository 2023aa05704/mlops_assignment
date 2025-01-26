import pandas as pd

data = pd.read_csv('data/housing.csv')
print(data.shape)

# Encode binary categories
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
data[binary_cols] = data[binary_cols].replace({'yes': 1, 'no': 0})

# Save intermediate file
data.to_csv("data/housing_pre.csv", index=False)