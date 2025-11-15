# tests/test_preprocessing.py
import pytest
from src.preprocessing import load_data, preprocessing

def test_preprocessing_returns_non_empty():
    df = load_data("data/covid.xlsx")
    X, y = preprocessing(df)
    
    assert X.shape[0] > 500   # on veut au moins ~600 lignes
    assert X.shape[1] > 10    # au moins quelques features
    assert len(y.unique()) == 2
    assert y.dtype == int

def test_est_malade_feature_exists():
    df = load_data("data/covid.xlsx")
    df = preprocessing(df)[0]  # on prend juste X
    assert 'est_malade' in df.columns