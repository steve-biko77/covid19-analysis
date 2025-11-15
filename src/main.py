import os
import sys

# Ajoute la racine du projet au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_data, preprocessing
from src.models import train_best_svm
from src.utils import evaluation
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print("=== CHARGEMENT DES DONNÉES ===")
    df = load_data()
    
    print("\n=== PREPROCESSING (dropna brut, comme tu veux) ===")
    X, y = preprocessing(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    
    print("\n=== ENTRAÎNEMENT SVM OPTIMISÉ ===")
    best_model = train_best_svm(X_train, y_train)
    
    print("\n=== ÉVALUATION FINALE ===")
    evaluation(best_model, X_train, X_test, y_train, y_test)