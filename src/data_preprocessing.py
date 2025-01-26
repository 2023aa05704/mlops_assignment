import pandas as pd

data = pd.read_csv('data/housing.csv')
print(data.shape)

# Encode binary categories
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
data[binary_cols] = data[binary_cols].replace({'yes': 1, 'no': 0})

# One-hot encode furnishingstatus
data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)
print(data.columns)
data['furnishingstatus_semi-furnished'] = data['furnishingstatus_semi-furnished'].replace({True: 1, False: 0})
data['furnishingstatus_unfurnished'] = data['furnishingstatus_unfurnished'].replace({True: 1, False: 0})

print(data.head())

# Save intermediate file
data.to_csv("data/housing_pre.csv", index=False)