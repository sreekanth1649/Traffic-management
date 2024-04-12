import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'ID': [1, 2, 3, 4, 5],
    'Junction': ['A', 'B', 'A', 'C', 'B'],
    'Date time': ['2022-01-01 08:00:00', '2022-01-01 12:00:00', '2022-01-02 09:00:00', '2022-01-02 14:00:00', '2022-01-03 10:00:00'],
    # Add other relevant columns in your dataset
    'Target_Variable': [10, 15, 8, 20, 12]  # Replace 'Target_Variable' with the actual variable you want to predict
}

# Create DataFrame
traffic_data = pd.DataFrame(data)

# Extract features and labels
features = traffic_data.drop(['ID', 'Target_Variable'], axis=1)
labels = traffic_data['Target_Variable']

# Identify categorical columns for one-hot encoding
categorical_features = features.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns with handle_unknown='ignore'
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Apply the preprocessing to the features
features_encoded = preprocessor.fit_transform(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, labels, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
