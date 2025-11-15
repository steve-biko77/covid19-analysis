    # tests/test_models.py
from src.models import train_best_svm
from src.preprocessing import load_data, preprocessing

def test_model_trains_without_error():
    df = load_data("data/covid.xlsx")
    X, y = preprocessing(df)
    
    model = train_best_svm(X, y)
    score = model.score(X, y)
    
    assert score > 0.6  # au moins 70% accuracy (facile avec ce dataset)