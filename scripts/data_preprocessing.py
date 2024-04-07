import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTENC

def preprocess_data(df):
    # Define predefined mapping for 'Failure Type' column
    cause_dict = {'No Failure': 0, 'PWF': 1, 'OSF': 2, 'HDF': 3, 'TWF': 4} 
    
    # Mapping the 'Failure Type' column using the predefined dictionary
    df['Failure Type'] = df['Failure Type'].map(cause_dict)
    
    # Apply One-hot encoding to the 'Type' column
    encoder = OneHotEncoder(drop='first')
    type_encoded = encoder.fit_transform(df[['Type']]).toarray()
    type_columns = encoder.get_feature_names_out(['Type'])
    type_encoded_df = pd.DataFrame(type_encoded, columns=type_columns)
    
    # Drop the original 'Type' column from the DataFrame
    df_encoded = df.drop(columns=['Type'])
    
    # Concatenate the one-hot encoded columns with the DataFrame
    df_encoded = pd.concat([df_encoded, type_encoded_df], axis=1)

    # Convert boolean values to integers (0 and 1) for specific columns
    df_encoded[['Type_H', 'Type_L', 'Type_M']] = df_encoded[['Type_H', 'Type_L', 'Type_M']].astype(int)

    # Feature engineering
    df_encoded['Temperature_difference'] = df_encoded['Air temperature [K]'] - df_encoded['Process temperature [K]']
    df_encoded['Power_output'] = df_encoded['Torque [Nm]'] * df_encoded['Rotational speed [rpm]']

    # Scale numerical features using StandardScaler
    scaler = StandardScaler()
    numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                          'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Temperature_difference', 'Power_output']
    df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

    return df_encoded

def split_data(df):
    X = df.drop(columns=['Machine failure', 'Failure Type'])
    y = df[['Machine failure', 'Failure Type']]
    
    # Split data into features and targets
    X, y = df.drop(columns=['Machine failure', 'Failure Type']), df[['Machine failure', 'Failure Type']]
    
    # Split data into training, validation, and testing sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=df['Failure Type'], random_state=0)
    X_train, X_val, y_trainval, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11, stratify=y_trainval['Failure Type'], random_state=0)

    return X_train, X_val, X_test, y_train, y_val, y_test

def handle_imbalanced_data(X_train, y_train):
    # Identify categorical features
    categorical_features = ['Type']

    # Use SMOTENC to handle class imbalance
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=0)
    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(r"C:\Users\USER\Documents\Python Scripts\Machine learning\Neural networks\Predictive Maintenance\predictive-maintenance-supervised-learning\data\ai4i2020.csv")

    # Preprocess data and split into training, validation, and testing sets
    df_preprocessed = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_preprocessed)
    
    # Handle imbalanced data
    X_train_resampled, y_train_resampled = handle_imbalanced_data(X_train, y_train)
