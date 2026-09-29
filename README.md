# BAAC — Classification de la gravité des accidents routiers

Projet de Machine Learning basé sur les données BAAC (Bulletins d’Analyse des Accidents Corporels), portant sur les accidents corporels de la circulation routière en France entre 2021 et 2024.

## Objectif

L’objectif du projet est d’étudier les facteurs associés à la gravité des accidents et de construire un modèle de classification binaire permettant d’identifier les usagers impliqués dans un accident grave.

La variable cible `grav` est transformée en deux classes :

- `0` : non grave — indemne ou blessé léger (`grav` original `1` et `4`)
- `1` : grave — tué ou blessé hospitalisé (`grav` original `2` et `3`)

Le problème est donc formulé comme une classification binaire au niveau de l’usager.

## Données

Les données proviennent des fichiers BAAC publiés par l’Observatoire National Interministériel de la Sécurité Routière (ONISR).

Le projet exploite les données 2021 à 2024, soit quatre années et quatre tables par année :

| Table | Contenu |
|---|---|
| `carac` | caractéristiques générales de l’accident |
| `lieux` | informations relatives au lieu et à l’infrastructure |
| `usagers` | informations sur les usagers impliqués |
| `vehicules` | informations sur les véhicules impliqués |

La clé `Num_Acc` permet de relier les différentes tables. Les données sont finalement construites à la granularité **un usager par ligne**, car la variable `grav` est définie au niveau individuel.

## Préparation des données

La préparation comprend plusieurs étapes :

- harmonisation des structures et des noms de colonnes entre 2021 et 2024 ;
- concaténation des données annuelles ;
- contrôle de la granularité et des doublons ;
- traitement des valeurs codées `-1` comme valeurs non renseignées ;
- analyse des valeurs manquantes et suppression des lignes inexploitables restantes ;
- agrégation de certaines informations de la table `lieux` par `Num_Acc` ;
- création de variables regroupées pour réduire le nombre de modalités ;
- suppression de variables devenues redondantes ou non retenues pour la modélisation.

### Variables finales

Le jeu de données utilisé pour le Machine Learning contient **479 684 observations** et 11 colonnes, dont la cible.

| Variable | Rôle |
|---|---|
| `grav` | cible binaire |
| `catu` | catégorie d’usager |
| `sexe` | sexe de l’usager |
| `lum` | conditions d’éclairage |
| `agg` | localisation en agglomération |
| `atm` | conditions atmosphériques |
| `surf` | état de la surface de la chaussée |
| `catv_grp` | catégorie regroupée de véhicule |
| `issecu` | présence d’un équipement de sécurité |
| `age_grp` | tranche d’âge |
| `vma_grp` | tranche de vitesse maximale autorisée |

Les variables `age` et `vma` ont notamment servi à construire leurs versions regroupées avant d’être retirées de la base finale. La variable `an` a également été retirée de la base finale utilisée pour la modélisation.

## Analyse exploratoire

L’EDA porte sur :

- la distribution de la variable cible ;
- l’analyse univariée des variables usagers et environnementales ;
- les relations entre les variables explicatives et la gravité ;
- plusieurs analyses multivariées ;
- les associations entre variables catégorielles avec le V de Cramér ;
- l’identification de variables redondantes.

### Principaux constats

La cible binaire est déséquilibrée :

- classe `0` : environ **82 %**
- classe `1` : environ **18 %**

Plusieurs variables présentent des différences de répartition selon la gravité. Les analyses mettent notamment en évidence des différences selon la catégorie d’usager, la catégorie de véhicule, l’âge et le port d’un équipement de sécurité.

L’analyse des associations entre variables met notamment en évidence une forte association entre `agg` et `vma_grp` avec un **V de Cramér de 0,85**.

## Modélisation

Le workflow Machine Learning comprend :

1. séparation de `X` et `y` ;
2. division train/test avec stratification de la cible ;
3. encodage des variables catégorielles avec `OneHotEncoder` pour les modèles scikit-learn ;
4. utilisation d’un `ColumnTransformer` et de `Pipeline` ;
5. prise en compte du déséquilibre des classes avec `class_weight='balanced'` lorsque le modèle le permet ;
6. comparaison de plusieurs modèles ;
7. optimisation de certains hyperparamètres ;
8. évaluation sur le jeu de test avec plusieurs métriques.

### Modèles testés

- Dummy Classifier — baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- CatBoost

## Résultats

Les performances sont principalement analysées sur la **classe 1**, correspondant aux accidents graves.

| Modèle | Accuracy | Precision classe 1 | Recall classe 1 | F1 classe 1 |
|---|---:|---:|---:|---:|
| Baseline | 0,82 | 0,00 | 0,00 | 0,00 |
| Logistic Regression | 0,68 | 0,33 | 0,73 | 0,45 |
| Decision Tree | 0,69 | 0,33 | 0,72 | 0,45 |
| Random Forest optimisé | 0,69 | 0,34 | 0,75 | 0,46 |
| Gradient Boosting | 0,83 | 0,58 | 0,19 | 0,29 |
| **CatBoost** | **0,69** | **0,33** | **0,76** | **0,46** |

La baseline atteint 82 % d’accuracy en prédisant la classe majoritaire, mais ne détecte aucun accident grave.

La régression logistique équilibrée améliore fortement la détection de la classe 1, avec 73 % de recall. Son optimisation par GridSearchCV puis Optuna n’apporte cependant pas d’amélioration significative du F1-score.

Le Random Forest optimisé avec Optuna améliore légèrement le recall à 75 % et atteint un F1-score de 0,46.

CatBoost obtient les meilleures performances de recall parmi les modèles testés, avec **76 % de recall** et **0,46 de F1-score** pour la classe grave.

## Optimisation

### Logistic Regression

GridSearchCV sélectionne `C = 10`, avec un F1-score moyen de 0,453.

Optuna converge vers une valeur de `C ≈ 9,37`, pour un F1-score moyen de 0,4526.

Ces résultats étant pratiquement identiques, l’hyperparamètre `C` n’apporte pas d’amélioration significative.

### Random Forest

Optuna sélectionne les paramètres suivants :

```text
n_estimators = 245
max_depth = 26
min_samples_split = 7
min_samples_leaf = 6
max_features = sqrt
class_weight = balanced
```

Le meilleur F1-score obtenu pendant cette optimisation est d’environ **0,463**.

## Modèle final : CatBoost

CatBoost est retenu comme modèle final principalement en raison de son **recall de 76 %** sur la classe grave.

Sur le jeu de test :

- Accuracy : **69 %**
- Precision classe 1 : **33 %**
- Recall classe 1 : **76 %**
- F1-score classe 1 : **0,46**

La matrice de confusion obtenue est :

```text
[[52840, 25872],
 [ 4218, 13007]]
```

Le modèle identifie donc **13 007 cas graves** parmi les observations réellement graves, contre **4 218 cas graves non détectés**.

En contrepartie, le nombre de faux positifs est élevé : **25 872 usagers non graves sont prédits comme graves**. Cela explique la precision limitée à 33 %.

Le choix de CatBoost constitue ainsi un compromis entre la détection des accidents graves et le nombre de fausses alertes.

## Limites

Les performances obtenues montrent que la prédiction de la gravité reste difficile avec les variables retenues.

Le modèle final présente un recall relativement élevé, mais une precision faible et un F1-score modéré. Le nombre important de faux positifs limite donc la fiabilité des prédictions positives.

Une autre limite concerne la procédure d’optimisation utilisée dans le notebook : les fonctions Optuna du projet évaluent directement les essais sur `X_test`. Dans une démarche de production ou d’évaluation expérimentale stricte, un jeu de validation ou une validation croisée devrait être utilisé pour l’optimisation, en conservant `X_test` uniquement pour l’évaluation finale.

## Structure du projet

```text
classif_BAAC/
│
├── data/
│   ├── raw/                  # Données BAAC brutes
│   └── processed/            # Dataset final préparé pour le ML
│
├── notebooks/
│   ├── EDA.ipynb
│   └── modelisation.ipynb
│
├── src/                      # Code source réutilisable
│
├── environment.yml           # Environnement Conda
├── .gitignore
└── README.md
```

## Environnement

Le projet est réalisé avec Python et un environnement Conda dédié.

Activation :

```bash
conda activate CLASSIF_BAAC
```

Les principales bibliothèques utilisées sont notamment :

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
optuna
catboost
```

## Reproduire le projet

1. Placer les données BAAC brutes dans `data/raw/`.
2. Exécuter le notebook d’EDA afin de construire le dataset préparé.
3. Générer `data/processed/baac_final.csv`.
4. Exécuter le notebook de modélisation.
5. Comparer les modèles à partir du recall, de la precision, du F1-score et des matrices de confusion.

