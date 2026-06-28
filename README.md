# Apprentissage Statistique Régularisé pour la Prévision des Rendements Boursiers

**PFE (Projet de Fin d'Études)** - Licence Sciences et Techniques, Mathématiques-Informatique

**Université** : Abdelmalek Essaâdi (Faculté des Sciences et Techniques d'Al Hoceima)  
**Auteurs** : ESSADDIKI Mohamed & AZARYAH Omar  
**Encadrant** : Prof. Abdelatif HAFID  
**Année** : 2025--2026  
**Soutenance** : 24 juin 2026

---

## 📋 Description du Projet

Ce projet implémente une **chaîne complète d'apprentissage statistique régularisé** pour la prévision du cours de clôture d'Apple (AAPL) en utilisant les données historiques de 2012 à 2019.

### Objectifs
- ✅ Comparer des modèles de régression régularisée (Ridge, Lasso, Elastic Net)
- ✅ Évaluer la parcimonie (sparsité) introduite par L1
- ✅ Confronter avec un modèle non-linéaire (Random Forest)
- ✅ Analyser critiquement les résultats et limitations

### Résultats Clés
| Modèle | RMSE | MAE | R² |
|--------|------|-----|-----|
| **OLS** | 0.516 | 0.398 | 0.996 |
| **Ridge (L2)** | 0.698 | 0.505 | 0.992 |
| **Lasso (L1)** | 0.478 | 0.371 | 0.996 |
| **Elastic Net** | 0.477 | 0.371 | 0.996 |
| **Random Forest** | 14.281 | 11.905 | -2.304 |

**Lasso sélectionne uniquement 37 variables actives sur 420**, démontrant efficacement la parcimonie.

---

## 🗂️ Architecture du Projet

```
├── stock_prediction_pfe.py          # Script principal (entraînement et évaluation)
├── requirements.txt                 # Dépendances Python
├── README.md                        # Vous êtes ici
└── resultats_test.csv              # Résultats de performance (généré après exécution)
```

---

## 🚀 Installation et Utilisation

### Prérequis
- Python 3.8+
- pip

### Étapes d'installation

1. **Cloner le repository** :
```bash
git clone https://github.com/your-username/stock-prediction-pfe.git
cd stock-prediction-pfe
```

2. **Créer un environnement virtuel** (optionnel mais recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

4. **Lancer le script** :
```bash
python stock_prediction_pfe.py
```

Le script téléchargera automatiquement les données historiques d'AAPL via **yfinance** et générera :
- Graphiques de prédictions pour chaque modèle
- Fichier `resultats_test.csv` avec les métriques de performance

---

## 📊 Méthodologie

### 1. **Données**
- **Source** : yfinance (données historiques AAPL)
- **Période** : 2012-01-01 à 2019-12-31
- **Variables OHLCV** : Close, High, Low, Open, Volume
- **Indicateurs techniques** : RSI (14), MACD (12, 26, 9)
- **Normalization** : MinMax (0, 1)

### 2. **Prétraitement**
- Construction de **séquences glissantes** (sliding windows) de **60 jours**
- Aplatissement pour les modèles linéaires : **60 × 7 = 420 variables**
- Split train/test : **80/20**

### 3. **Modèles Testés**

#### A. Régression Linéaire Ordinaire (OLS)
Modèle de référence, sans pénalité.

#### B. Ridge (L2)
Pénalité L2 : $\lambda \sum_{j=1}^{p} \beta_j^2$

Hyperparamètre tuné : `alpha ∈ [10⁻⁴, 10²]` (30 valeurs)

#### C. Lasso (L1)
Pénalité L1 : $\lambda \sum_{j=1}^{p} |\beta_j|$

**Résultat** : 383 coefficients annulés sur 420 (91% sparsité)

Hyperparamètre tuné : `alpha ∈ [10⁻⁵, 10⁻¹]` (30 valeurs)

#### D. Elastic Net
Combinaison L1 + L2 : $\lambda [ (1 - \alpha) \sum |\beta_j| + \alpha \sum \beta_j^2 ]$

Hyperparamètres tunés : 
- `alpha ∈ [10⁻⁵, 10⁻¹]` (20 valeurs)
- `l1_ratio ∈ {0.2, 0.5, 0.8}` (3 valeurs)

#### E. Random Forest
- `n_estimators=100`
- `max_depth=12`
- Comparaison non-linéaire (serve as counter-example)

### 4. **Validation**
- **Time Series Split** : 5 folds (respecte l'ordre temporel)
- **Métrique de tuning** : neg_mean_squared_error
- **Parallélisation** : n_jobs=-1

### 5. **Métriques d'Évaluation**
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

---

## 🔍 Analyse Critique

### Forces
1. ✅ **Parcimonie (Lasso)** : Seules 37 variables pertinentes identifiées
2. ✅ **Performance (Elastic Net)** : Meilleure RMSE (0.477 USD)
3. ✅ **Méthodologie rigoureuse** : Time Series Split, validation temporelle

### Limitations
1. ⚠️ **Cible en niveau de prix** : Série tendancielle, non rendements
2. ⚠️ **Fuite mineure du scaler global** : Ajusté avant split (future information leak)
3. ⚠️ **Un seul actif, une seule période** : Robustesse limitée
4. ⚠️ **Pas de dimension économique** : Aucun coût de transaction, slippage, ou mesure de risque
5. ⚠️ **Random Forest s'effondre** : R² = -2.30 (pas de capacité d'extrapolation)

### Leçon Majeure
> **Un R² de 0.996 ne prouve pas un modèle génial, mais plutôt une cible facile à prédire** (persistance du prix). En finance quantitative, **savoir lire les résultats avec recul est aussi important que savoir les produire**.

---

## 🔧 Extensions Futures

1. **Reformuler en rendements** : Cible réelle = signal faible noyé dans le bruit
2. **Durcir le protocole** : Scaler entraînement seul, walk-forward complet
3. **Étendre le champ** : Autres actifs (MASI - Bourse de Casablanca), autres périodes
4. **Enrichir les variables** : Modèles séquentiels (ARIMA, LSTM) avec respect strict du protocole temporel
5. **Franchir au réel** : Backtest complet (coûts, slippage, risk management)

---

## 📚 Références Clés

- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso
- Zou, H. & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net
- Hoerl, A. E. & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems
- Breiman, L. (2001). Random Forests
- Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning

---

## 📄 Licence

Ce projet est soumis aux règles de l'Université Abdelmalek Essaâdi et est réservé à des fins académiques.

---

## 📧 Contact

Pour des questions ou clarifications :
- **Encadrant** : Prof. Abdelatif HAFID
- **Jury** : 
  - Présidente : Pr. Najat MORADI
  - Examinateur : Pr. Omar DARHOUCHE

---

**Dernière mise à jour** : Juin 2026
