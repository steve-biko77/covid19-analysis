# Rapport EDA : Analyse Exploratoire des Données COVID-19

## Objectif
- Comprendre au mieux les données (petit pas en avant vaut mieux qu'un grand pas en arrière).
- Développer une première stratégie de modélisation.

## Checklist de Base
### 1) Analyse de Forme
- **Target** : SARS-Cov-2 exam result.
- **Lignes et colonnes** : 5644 rows × 111 columns.
- **Types de variables** : float64 (70), object (36), int64 (5).
- **Valeurs manquantes** : Beaucoup de NaN (moitié des variables > 90%). Deux groupes : 
  - Tests viraux (~76% NaN).
  - Taux sanguins (~89% NaN).
- Visualisations : `sns.heatmap(data.isna())`, `(data.isna().sum()/data.shape[0]).sort_values(ascending=True)`.

### 2) Analyse du Fond
- Éliminer colonnes inutiles (>90% NaN ou sans lien target) : `data[data.columns[(data.isna().sum()/data.shape[0]) < 0.9]]`.
- Afficher heatmap NaN avant analyse : `plt.figure(figsize=(20,10)); sns.heatmap(data.isna())`.

#### 2.1) Visualisation de la Target
- `data['SARS-Cov-2 exam result'].value_counts(normalize=True)` : 10% de cas positif.

#### 2.2) Signification des Variables
- **Continues** (histogrammes) :
  - Variables continues standardisées, skewed/asymétriques ; tests sanguins : `sns.distplot(data[col])`.
  - Age quantile : Difficile à interpréter (données traitées, 0-5 ou transformation mathématique ? Pas précisé dans dataset, pas critique).
- **Catégoriques** (histogrammes) :
  - Qualitatives binaires (0/1) ; viraux, Rhinovirus élevé.

#### 2.3) Relations Variables / Target
- Création sous-ensembles positifs/négatifs (ou autres catégories).
- Target/Blood : Taux Monocytes, Platelets, Leukocytes semblent liés au COVID-19 → hypothèses à tester (`distplot`).
- Target/Age : Difficile à prononcer ; faibles âges peu contaminés ? Attention : âge inconnu, date dataset inconnue (enfants touchés autant qu'adultes). Variable intéressante pour comparer avec tests sanguins.
- Target/Viral (`crosstab`) : Doubles maladies rares. Rhinovirus/Enterovirus positif - COVID négatif → hypothèse à tester (épidémie régionale ? Possible 2 virus en même temps, pas lien direct COVID).

## Conclusion Initiale
- Beaucoup de données manquantes (au mieux 20% gardé).
- 2 groupes intéressants (viral, sanguin).
- Presque pas de variable discriminante pour positif/négatif → pas approprié prédire COVID avec simples tests sanguins. Mais poursuivre analyse pour apprendre. ML ne transforme pas eau en vin, mais pas raison abandonner. Bon data scientist va au bout ; démontrer robustement si histoire pas simple.
- Positif : Variables intéressantes identifiées pour rôle non négligeable.

## Analyse Avancée
### 1.1) Relations Variables/Variables
- **Taux Sanguins** (`sns.heatmap(data[blood_columns].corr())`, `sns.clustermap(data[blood_columns].corr())`, `sns.pairplot(data[blood_columns])`) :
  - Blood/Blood : Certaines variables très corrélées (0.9, surveiller plus tard).
  - Blood/Age (`lmplot`) : Très faible corrélation.
- **Viral/Viral** : Influenza rapid test donne mauvais résultats → peut-être drop.
- **Maladie/Blood** : Taux sanguins entre malades et COVID diffèrent.
- **Hospitalisation/Est Malade**.
- **Hospitalisation/Test Sanguin** : Intéressant pour prédire service patient.

### 1.2) Analyse des NaN
- `data[viral_columns].count()`, `data[blood_columns].count()`, `data.dropna().count()`.
- Test possibilités : Drop all NaN → 99 éléments/colonnes sur 5000.
- Dataset distingué par 2 groupes (blood_columns, viral_columns).
- Proportions : Choisir un des deux, rapports target différents :
  - Viral : 1350 (92/8).
  - Blood : 600 (87/13).
  - Both : 90.
- Tester hypothèses avec t-test de Student.