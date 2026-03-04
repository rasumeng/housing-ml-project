import pandas as pd
from sklearn.impute import SimpleImputer

def impute_missing(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df