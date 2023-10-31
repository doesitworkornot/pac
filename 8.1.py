import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/wells_info_with_prod.csv')
X = pd.DataFrame(columns=['water', 'len'])
y = df.Prod1Year
X.water = df.WATER_PER_FOOT
X.len = df.LATERAL_LENGTH_BLEND

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()

features_train_scaled = scaler.fit_transform(X_train)
features_test_scaled = scaler.transform(X_test)

target_var_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
target_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))
