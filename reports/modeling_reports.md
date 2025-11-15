# Rapport Modeling : Prédiction COVID-19

## Objectif
Mettre les données en format propice au ML et améliorer la performance du modèle.
1. **Format ML** :
   - Train/Test : `train_test_split` (80/20, random_state=42).
   - Encodage : Map Pandas avec code `{'positive':1, 'negative':0, 'detected':1, 'not_detected':0}`.
   - Fonctions modulaires : `encodage(df)`, `feature_engineering(df)` ('est_malade' comme sum viraux >=1), `imputation(df)` (médiane via SimpleImputer pour éviter 0 lignes).
   - Preprocessing global : `def preprocessing(df)` → encodage + engineering (drop viraux) + imputation + X/y.
   - Nettoyage NaN : Imputation médiane (changement de logique vs. drop initial pour gérer overfitting).

2. **Améliorer performance** :
   - Feature Selection : SelectKBest (f_classif).
   - Feature Engineering : 'est_malade'.
   - Feature Scaling : StandardScaler dans pipeline.
   - Suppression outliers : Non appliquée dans version finale (optionnelle via Z-score si besoin).

## Premier Modèle & Diagnostic
- Modèles testés : RandomForest (pour overfitting), AdaBoost, SVM (meilleur score), KNN.
- Évaluation : 
  - Confusion matrix, classification report, F1-score.
  - Learning curve (F1, cv=4, train_sizes linspace 0.1-1).
- Boucle itérative : Idée → Code → Éval.
  - Overfitting initial détecté via learning curve (train haut, val bas).
  - Changement logique imputation : De dropna à médiane → amélioration (plus de lignes gardées ~600).
  - Éliminer viraux : Fait dans engineering → pas d'amélioration majeure.
  - Modèle approprié overfitting : RandomForest testé ; SVM meilleur global.

## Optimisation
- **SVM**  : RandomizedSearchCV (scoring='recall', cv=4, n_iter=40).
  - Params : gamma [1e-3,1e-4], C [1,10,100,1000], degree [2,3,4], k range(40,60).
- **AdaBoost**  : RandomizedSearchCV similaire.
  - Params : n_estimators range(120,200), learning_rate [0.1-0.7], degree [2,3,4], k range(30,80).
- Precision-Recall curve pour threshold custom.
- Model final : Decision function > threshold (-1 testé) ; F1 et recall calculés.

## Résultats Clés (à remplir avec tes runs finaux)
- Exemple SVM optimisé : F1 ~0.56, Recall ~0.56(positif classe minoritaire).
- Learning curve : Amélioration post-itérations, mais overfitting persistant (val score stable bas).
- Meilleur modèle : SVM (atteint meilleures performances) ; Ada pour comparaison (n'atteint pas SVM).

## Conclusions
- Overfitting géré via itérations (imputation médiane, drop viraux, RandomForest/AdaBoost).
- Pas réelle amélioration massive learning curve → limites dataset (NaN, imbalance).
- Améliorations efficaces : Selection/Engineering/Scaling ; outliers optionnel.
- Performance : F1 non négligeable ; boucle itérative démontre robustesse. SVM meilleur ; Ada testé pour comparaison (n'atteint pas ses performances).
- Suggestions futures : SMOTE pour imbalance, plus data, ou focus hospitalisation comme target alternatif.