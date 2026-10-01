import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

trip_data = {
    'distance_km': [14.0, 42.0, 9.5, 27.0, 58.0, 18.0, 31.5, 7.0],
    'traffic_score': [2, 5, 1, 3, 4, 2, 4, 1],
    'cargo_weight_kg': [30, 210, 15, 95, 340, 45, 120, 10],
    'trip_time_minutes': [32, 105, 18, 62, 138, 41, 80, 16]
}

df = pd.DataFrame(trip_data)

features = ['distance_km', 'traffic_score', 'cargo_weight_kg']
target = 'trip_time_minutes'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

predicted_times = regressor.predict(X_test)
print("Actual delivery times:", list(y_test))
print("Predicted delivery times:", [round(val, 1) for val in predicted_times])
