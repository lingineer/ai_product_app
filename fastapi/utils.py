import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def prep_gender(df):
    # preprocess the 
    mapping = {
        'Female': 0,
        'Male': 1
    }
    df = df.copy()
    df['gender'] = df['gender'].map(mapping).astype(int)
    return df

def prep_customer_status(df):
    mapping = {
        'Churned': 1,
        'Stayed': 0
    }
    df = df.copy()
    if 'status' in df.columns:
        df['status'] = df['status'].map(mapping).astype(int)
    return df

def prep_payment_method(df, encoder=None):
    df = df.copy()
    cols = ['payment_method']
    df_target = df[cols]
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(df_target)
    encoded = encoder.transform(df_target)
    df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cols))
    df_encoded = df_encoded.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df_transformed = pd.concat([df, df_encoded], axis=1)
    df_transformed = df_transformed.drop(cols, axis=1)
    return df_transformed, encoder

def scale_num_cols(df, scaler=None):
    df = df.copy()
    cols = ['age', 'download', 'charge']
    df_target = df[cols]
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df_target)
    df_target = scaler.transform(df_target)
    df[cols] = df_target
    return df, scaler

def impute_missing(df, imputer=None):
    df = df.copy()
    cols = ['download']
    df_target = df[cols]
    if imputer is None:
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df_target)
    df_target = imputer.transform(df_target)
    df[cols] = df_target
    return df, imputer
    

def preprocessing(df, artifacts={}):
    df = df.copy()
    df = prep_gender(df)
    df = prep_customer_status(df)
    if not artifacts:
        df, encoder = prep_payment_method(df)
        df, scaler = scale_num_cols(df)
        df, imputer = impute_missing(df)
        artifacts = {
            'encoder': encoder,
            'scaler': scaler,
            'imputer': imputer
        }
    else:
        df, _ = prep_payment_method(df, artifacts['encoder'])
        df, _ = scale_num_cols(df, artifacts['scaler'])
        df, _ = impute_missing(df, artifacts['imputer'])
    return df, artifacts
