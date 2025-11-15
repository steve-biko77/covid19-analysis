# src/utils.py
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def evaluation(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print("           RÉSULTATS FINAUX - SVM OPTIMISÉ")
    print("="*50)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"F1-score final : {f1_score(y_test, y_pred):.4f}")
    
    # Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=4, scoring='f1', train_sizes=np.linspace(0.3, 1.0, 6)
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Score d'entraînement", marker='o')
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Score de validation", marker='s')
    plt.title("Courbe d'apprentissage - SVM (F1-score)")
    plt.xlabel("Nombre d'exemples d'entraînement")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()