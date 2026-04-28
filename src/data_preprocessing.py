
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    # Ensure this line is indented exactly 4 spaces
    return pd.read_csv(path)

def preprocess(df):
    # All lines below must be indented exactly 4 spaces
    df = df.replace(" ", np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.drop('customerID', axis=1)
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['Churn'])
    df = df.drop('Churn', axis=1)
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    return X, y