import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib  # Using joblib instead of pickle

# Load and preprocess your data
# ... (your existing preprocessing code)

# Create and train model
# ... (your existing model code)

# Save with joblib
joblib.dump(model, 'pipe.pkl')
joblib.dump(df, 'df.pkl')
