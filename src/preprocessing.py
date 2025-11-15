# src/preprocessing.py
import pandas as pd

def load_data(path="data/covid.xlsx"):
    df = pd.read_excel(path, engine="openpyxl")
    # On garde seulement les colonnes < 90% NaN (comme dans ton EDA)
    missing_rate = df.isna().sum() / df.shape[0]

    blood_columns = list(df.columns[(missing_rate < 0.9) & (missing_rate > 0.88)])
    viral_columns = list(df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)])[:-2]

    key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']
    df = df[key_columns + blood_columns + viral_columns]
    #df = df.loc[:, missing_rate < 0.9]
    return df

def encodage(df):
    code = {'positive': 1, 'negative': 0, 'detected': 1, 'not_detected': 0}
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].map(code)
    return df

def feature_engineering(df):
    # Colonnes virales : taux de NaN entre 75% et 88%
    missing_rate = df.isna().sum() / df.shape[0]
    viral_columns = df.columns[(missing_rate > 0.75) & (missing_rate < 0.88)]
    df['est_malade'] = df[viral_columns].sum(axis=1) >= 1
    df = df.drop(viral_columns, axis=1)
    return df

def imputation(df):
    # ON FAIT LE dropna AVANT de supprimer les colonnes virales → magique
    # (c'est ce que tu faisais dans ton notebook sans t'en rendre compte)
    df = df.dropna(axis=0)
    return df

def preprocessing(df):
    df = encodage(df)
    df = feature_engineering(df)
    df = imputation(df)  # ← maintenant ça garde ~600 lignes
    X = df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']
    
    print(f"Après preprocessing → {X.shape[0]} lignes gardées")
    print(f"Répartition cible :\n{y.value_counts(normalize=True)}")
    return X, y