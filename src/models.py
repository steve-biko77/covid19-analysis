from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

def train_best_svm(X_train, y_train):
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        SelectKBest(f_classif),
        StandardScaler(),
        SVC(random_state=0)
    )
    
    params = {
        'svc__C': [1, 10, 100, 1000],
        'svc__gamma': ['scale', 'auto', 0.001, 0.0001],
        'polynomialfeatures__degree': [2, 3],
        'selectkbest__k': range(40, 60)
    }
    
    grid = RandomizedSearchCV(model, params, scoring='f1', cv=4, n_iter=40, random_state=0, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Meilleurs paramètres :", grid.best_params_)
    return grid.best_estimator_