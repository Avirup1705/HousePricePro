import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# Load dataset
data = pd.read_csv("House Price Prediction Dataset.csv")

# Convert Location to numeric
data = pd.get_dummies(data, columns=["Location"])

# Features
X = data.drop(["Price","Id","Condition","Garage","Floors"], axis=1)

# Target
y = data["Price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

# Model
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))

print("Model trained and saved successfully")