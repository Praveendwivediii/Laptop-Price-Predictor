import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib  # Using joblib instead of pickle for better compatibility

# Sample data - replace with your actual data loading
data = {
    'Company': ['Dell', 'HP', 'Lenovo', 'Asus', 'Apple']*20,
    'TypeName': ['Ultrabook', 'Notebook', 'Gaming', '2 in 1', 'Workstation']*20,
    'Ram': [8, 16, 32, 8, 16]*20,
    'Weight': [1.2, 2.1, 2.5, 1.8, 3.0]*20,
    'Touchscreen': ['No', 'No', 'Yes', 'Yes', 'No']*20,
    'IPS': ['Yes', 'No', 'Yes', 'No', 'Yes']*20,
    'screen_size': [13.3, 15.6, 17.3, 14.0, 16.0]*20,
    'resolution_x': [1920, 1366, 1600, 3840, 3200]*20,
    'resolution_y': [1080, 768, 900, 2160, 1800]*20,
    'Cpu brand': ['Intel', 'AMD', 'Intel', 'AMD', 'Intel']*20,
    'HDD': [0, 256, 512, 0, 1024]*20,
    'SSD': [256, 0, 512, 1024, 0]*20,
    'Gpu brand': ['Nvidia', 'AMD', 'Nvidia', 'Intel', 'Nvidia']*20,
    'os': ['Windows', 'Windows', 'Linux', 'Mac', 'Windows']*20,
    'Price': [75000, 50000, 120000, 150000, 200000]*20
}

df = pd.DataFrame(data)

# Data preprocessing
df['Touchscreen'] = df['Touchscreen'].map({'Yes': 1, 'No': 0})
df['IPS'] = df['IPS'].map({'Yes': 1, 'No': 0})

# Calculate PPI
df['ppi'] = np.sqrt(df['resolution_x']**2 + df['resolution_y']**2)/df['screen_size']

# Features and target
X = df[['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS', 'ppi', 
        'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']]
y = np.log(df['Price'])  # Log transform for better performance

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = ['Ram', 'Weight', 'ppi', 'HDD', 'SSD']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['Company', 'TypeName', 'Cpu brand', 'Gpu brand', 'os']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline with model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Save the model and dataframe using joblib
joblib.dump(model, 'pipe.pkl')
joblib.dump(df, 'df.pkl')

print("Model and data saved successfully as pipe.pkl and df.pkl")
