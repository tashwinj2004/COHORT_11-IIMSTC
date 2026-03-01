import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. LOAD DATA
df = pd.read_csv('cleaned_fw_dataset.csv')

# 2. FEATURE SELECTION
# We use numeric columns and 'Category'. Dates are already reflected in 'Shelf_Life'.
features = ['Category', 'Stock_Quantity', 'Unit_Price', 'Sales_Volume', 
            'Inventory_Turnover_Rate', 'Shelf_Life']
target = 'Food_Waste_kg'

X = df[features]
y = df[target]

# 3. PREPROCESSING
# Convert the 'Category' column into numbers (One-Hot Encoding)
X = pd.get_dummies(X, columns=['Category'])

# 4. SPLIT DATA (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. BUILD & TRAIN THE MODEL
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. MAKE PREDICTIONS & EVALUATE
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Training Results:")
print(f"--- Mean Absolute Error: {mae:.4f} kg")
print(f"--- R-Squared (Accuracy): {r2:.4f}")

# 7. EXAMPLE PREDICTION
# To predict on new data, use: model.predict(your_new_data)