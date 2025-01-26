import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

# Save to file
data.to_csv("data/housing_pre.csv", index=False)

# Split into features (X) and target (y)
X = data.drop('price', axis=1)
y = data['price']

# Scale numeric features
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Save to file
data.to_csv("data/housing_pre.csv", index=False)

X_train.to_csv("data/processed_X_train.csv", index=False)
X_test.to_csv("data/processed_X_test.csv", index=False)
y_train.to_csv("data/processed_y_train.csv", index=False)
y_test.to_csv("data/processed_y_test.csv", index=False)