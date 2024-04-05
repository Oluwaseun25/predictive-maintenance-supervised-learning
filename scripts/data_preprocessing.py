import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def preprocess_data(df):
    # Handle missing values if any
    # No missing values based on data overview
    
    # Encode categorical variables
    encoder = OneHotEncoder(drop='first')
    df_encoded = pd.get_dummies(df, columns=['Type'], drop_first=True)
    # One-hot encode the 'Failure Type' column
    df_encoded = pd.get_dummies(df_encoded, columns=['Failure Type'], drop_first=True)
    
    # Feature engineering
    df_encoded['Temperature_difference'] = df_encoded['Air temperature [K]'] - df_encoded['Process temperature [K]']
    df_encoded['Power_output'] = df_encoded['Torque [Nm]'] * df_encoded['Rotational speed [rpm]']
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_features = ['UDI', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                          'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Temperature_difference', 'Power_output']
    df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])
    
    return df_encoded

def split_data(df):
    X = df.drop(columns=['Machine failure'])
    y = df['Machine failure']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def handle_imbalanced_data(X_train, y_train):
    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled

def preprocessing_pipeline(df):
    # Preprocess data
    df_preprocessed = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_preprocessed)
    
    # Handle imbalanced data
    X_train_resampled, y_train_resampled = handle_imbalanced_data(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(r"C:\Users\USER\Documents\Python Scripts\Machine learning\Neural networks\Predictive Maintenance\predictive-maintenance-supervised-learning\data\ai4i2020.csv")

    
    # Preprocess data and split into training and testing sets
    X_train, X_test, y_train, y_test = preprocessing_pipeline(df)
    
    
