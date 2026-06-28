#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
    Apprentissage Statistique Régularisé pour la Prévision 
    des Rendements Boursiers (Application à l'action AAPL)
    
    PFE - Projet de Fin d'Études
    Auteurs : [Onex], AZARYAH Omar
    Encadrant : Prof. Abdelatif HAFID
    Université Abdelmalek Essaâdi - Faculté des Sciences et Techniques d'Al Hoceima
    Année universitaire 2025--2026
================================================================================

Résumé :
    Ce script implémente une chaîne complète d'apprentissage statistique régularisé
    pour la prévision du cours de clôture d'Apple (AAPL) en utilisant les données
    historiques 2012--2019.
    
    Modèles testés :
    - Régression Linéaire Ordinaire (OLS) - référence
    - Ridge (pénalité L2)
    - Lasso (pénalité L1) - sélection de variables
    - Elastic Net (combinaison L1 + L2)
    - Random Forest - comparaison non-linéaire
    
    Métriques d'évaluation : RMSE, MAE, R²
    Validation : Time Series Split (5 folds)

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 12

# =====================================================================
#  CHARGEMENT DES DONNEES
# =====================================================================
print("=" * 80)
print("CHARGEMENT DES DONNEES HISTORIQUES AAPL")
print("=" * 80)

# Téléchargement depuis yfinance : AAPL, 2012-01-01 à 2019-12-31
start_date = "2012-01-01"
end_date = "2019-12-31"
ticker = "AAPL"

print(f"\nTéléchargement : {ticker} ({start_date} à {end_date})")
data = yf.download(ticker, start=start_date, end=end_date, progress=False)
print(f"Données téléchargées : {len(data)} jours de trading")
print(f"\nAperçu des données :")
print(data.head())
print(f"\nDimensions : {data.shape}")
print(f"Colonnes : {list(data.columns)}")

# =====================================================================
#  CONSTRUCTION DES VARIABLES OHLCV ENRICHIES + INDICATEURS TECHNIQUES
# =====================================================================
print("\n" + "=" * 80)
print("CONSTRUCTION DES VARIABLES ENRICHIES")
print("=" * 80)

dataset = data[['Close', 'High', 'Low', 'Open', 'Volume']].copy()
dataset.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Indicateur 1 : RSI (Relative Strength Index) - simple smoothing
def calculate_rsi(prices, period=14):
    """Calcul du RSI avec lissage simple (non Wilder)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Indicateur 2 : MACD (Moving Average Convergence Divergence)
def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcul du MACD et sa signal line."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

dataset['RSI'] = calculate_rsi(dataset['Close'], period=14)
macd_line, macd_signal, macd_hist = calculate_macd(dataset['Close'])
dataset['MACD'] = macd_line
dataset['MACD_Signal'] = macd_signal

# Supprimer les NaN (dus au calcul des indicateurs)
dataset = dataset.dropna()
print(f"Variables finales (après calcul indicateurs) : {dataset.shape[1]}")
print(f"Données restantes : {len(dataset)} jours")
print(f"\nAperçu des données enrichies :")
print(dataset.head())

features = ['Close', 'High', 'Low', 'Open', 'Volume', 'RSI', 'MACD', 'MACD_Signal']
print(f"\nFonctionnalités : {features}")

# =====================================================================
#  NORMALISATION MinMax (scaler global ajusté sur l'ensemble)
# =====================================================================
print("\n" + "=" * 80)
print("NORMALISATION DES DONNEES (MinMax Scaler)")
print("=" * 80)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset[features])
print(f"Données normalisées : shape = {scaled_data.shape}")
print(f"Plage : [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")

# =====================================================================
#  SPLIT TRAIN / TEST (80/20)
# =====================================================================
print("\n" + "=" * 80)
print("SPLIT TRAIN / TEST")
print("=" * 80)

training_data_len = int(len(scaled_data) * 0.8)
test_data_len = len(scaled_data) - training_data_len
print(f"Ensemble d'entraînement : {training_data_len} jours (80%)")
print(f"Ensemble de test : {test_data_len} jours (20%)")
print(f"Date de split : {dataset.index[training_data_len]}")

# =====================================================================
#  CONSTRUCTION DES SEQUENCES GLISSANTES (sliding windows, 60 jours)
# =====================================================================
print("\n" + "=" * 80)
print("CONSTRUCTION DES SEQUENCES GLISSANTES (60 jours)")
print("=" * 80)

# --- Ensemble d'entraînement ---
train_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, :])   # 60 jours x 7 variables
    y_train.append(train_data[i, 0])        # cible : Close du jour i
x_train, y_train = np.array(x_train), np.array(y_train)
print(f"Training shape: {x_train.shape}")   # (m, 60, 7)

# --- Ensemble de test (on remonte de 60 jours pour la 1re fenetre) ---
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, 0].values     # prix de cloture reels (USD)
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, :])
x_test = np.array(x_test)
print(f"Test data shape: {x_test.shape}")

# --- Mise a plat pour les modeles lineaires : (m, 60*7) = (m, 420) ---
x_train_2d = x_train.reshape(x_train.shape[0], -1)
x_test_2d = x_test.reshape(x_test.shape[0], -1)
print(f"Reshaped x_train for linear models: {x_train_2d.shape}")
print(f"Reshaped x_test for linear models: {x_test_2d.shape}")

# =====================================================================
#  FONCTION D'AFFICHAGE DES PREDICTIONS (denormalisation incluse)
# =====================================================================
def plot_predictions(model_name, y_true, y_pred,
                     training_data_len, data, scaler):
    """Denormalise les predictions et trace train/reel/predictions."""
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()

    # tableau factice pour inverser la transformation MinMax
    dummy = np.zeros((len(y_pred), data.shape[1]))
    dummy[:, 0] = y_pred.flatten()
    predictions_unscaled = scaler.inverse_transform(dummy)[:, 0]
    valid.loc[:, 'Predictions'] = predictions_unscaled

    plt.figure(figsize=(16, 8))
    plt.title(f'{model_name} - Predictions vs Reel')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Entrainement', 'Reel', 'Predictions'],
               loc='lower right')
    plt.show()

    rmse = np.sqrt(mean_squared_error(y_true, predictions_unscaled))
    mae = mean_absolute_error(y_true, predictions_unscaled)
    r2 = r2_score(y_true, predictions_unscaled)
    print(f"{model_name} -> RMSE = {rmse:.4f} | "
          f"MAE = {mae:.4f} | R2 = {r2:.4f}")
    return predictions_unscaled

# =====================================================================
#  (A) REGRESSION LINEAIRE ORDINAIRE (OLS) -- modele de reference
# =====================================================================
print("\n" + "=" * 80)
print("(A) REGRESSION LINEAIRE ORDINAIRE (OLS)")
print("=" * 80)
linear_model = LinearRegression()
linear_model.fit(x_train_2d, y_train)
linear_predictions_scaled = linear_model.predict(x_test_2d)
linear_predictions = plot_predictions('Linear Regression', y_test, linear_predictions_scaled,
                 training_data_len, dataset, scaler)

# =====================================================================
#  (B) REGRESSION RIDGE (penalite L2)
# =====================================================================
print("\n" + "=" * 80)
print("(B) REGRESSION RIDGE (penalite L2)")
print("=" * 80)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train_2d, y_train)
ridge_predictions_scaled = ridge_model.predict(x_test_2d)
ridge_predictions = plot_predictions('Ridge Regression', y_test, ridge_predictions_scaled,
                 training_data_len, dataset, scaler)

# =====================================================================
#  (C) REGRESSION LASSO (penalite L1)
# =====================================================================
print("\n" + "=" * 80)
print("(C) REGRESSION LASSO (penalite L1)")
print("=" * 80)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train_2d, y_train)
lasso_predictions_scaled = lasso_model.predict(x_test_2d)
lasso_predictions = plot_predictions('Lasso Regression', y_test, lasso_predictions_scaled,
                 training_data_len, dataset, scaler)

# =====================================================================
#  (D) ELASTIC NET (combinaison L1 + L2)
# =====================================================================
print("\n" + "=" * 80)
print("(D) ELASTIC NET (combinaison L1 + L2)")
print("=" * 80)
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net_model.fit(x_train_2d, y_train)
elastic_net_predictions_scaled = elastic_net_model.predict(x_test_2d)
elastic_net_predictions = plot_predictions('Elastic Net Regression', y_test,
                 elastic_net_predictions_scaled,
                 training_data_len, dataset, scaler)

# =====================================================================
#  REGLAGE DES HYPERPARAMETRES PAR VALIDATION CROISEE TEMPORELLE
# =====================================================================
print("\n" + "=" * 80)
print("REGLAGE DES HYPERPARAMETRES (Time Series Cross-Validation)")
print("=" * 80)
tscv = TimeSeriesSplit(n_splits=5)

# --- Reglage du Ridge ---
print("\nTuning Ridge...")
param_grid_ridge = {'alpha': np.logspace(-4, 2, 30)}
grid_search_ridge = GridSearchCV(Ridge(), param_grid_ridge, cv=tscv,
                                 scoring='neg_mean_squared_error',
                                 n_jobs=-1)
grid_search_ridge.fit(x_train_2d, y_train)
best_alpha_ridge = grid_search_ridge.best_params_['alpha']
print(f"Best Alpha for Ridge: {best_alpha_ridge:.4f}")
ridge_model_tuned = grid_search_ridge.best_estimator_

# --- Reglage du Lasso ---
print("\nTuning Lasso...")
param_grid_lasso = {'alpha': np.logspace(-5, -1, 30)}
grid_search_lasso = GridSearchCV(Lasso(max_iter=20000),
                                 param_grid_lasso, cv=tscv,
                                 scoring='neg_mean_squared_error',
                                 n_jobs=-1)
grid_search_lasso.fit(x_train_2d, y_train)
best_alpha_lasso = grid_search_lasso.best_params_['alpha']
print(f"Best Alpha for Lasso: {best_alpha_lasso:.6f}")
lasso_model_tuned = grid_search_lasso.best_estimator_

# Parcimonie : nombre de coefficients exactement nuls
n_zero = (np.abs(lasso_model_tuned.coef_) < 1e-10).sum()
print(f"Coefficients annules par le Lasso: "
      f"{n_zero} / {x_train_2d.shape[1]}")

# --- Reglage de l'Elastic Net ---
print("\nTuning Elastic Net...")
param_grid_enet = {
    'alpha': np.logspace(-5, -1, 20),
    'l1_ratio': [0.2, 0.5, 0.8],
}
grid_search_enet = GridSearchCV(ElasticNet(max_iter=20000),
                                param_grid_enet, cv=tscv,
                                scoring='neg_mean_squared_error',
                                n_jobs=-1)
grid_search_enet.fit(x_train_2d, y_train)
print(f"Best params Elastic Net: {grid_search_enet.best_params_}")
enet_model_tuned = grid_search_enet.best_estimator_

# =====================================================================
#  (E) MODELE DE COMPARAISON NON LINEAIRE : RANDOM FOREST
# =====================================================================
print("\n" + "=" * 80)
print("(E) RANDOM FOREST (comparaison non-linéaire)")
print("=" * 80)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=12,
                                 random_state=42, n_jobs=-1)
rf_model.fit(x_train_2d, y_train)
rf_predictions_scaled = rf_model.predict(x_test_2d)
rf_predictions = plot_predictions('Random Forest', y_test, rf_predictions_scaled,
                 training_data_len, dataset, scaler)

# Importance agregee des variables (somme sur les 60 pas de temps)
importances = rf_model.feature_importances_.reshape(60, len(features))
agg_importance = importances.sum(axis=0)
print("\nImportance des variables (agrégée sur 60 timesteps) :")
for f, imp in zip(features, agg_importance):
    print(f"{f:10s} : {imp:.4f}")

# =====================================================================
#  TABLEAU RECAPITULATIF DES PERFORMANCES SUR LE TEST
# =====================================================================
print("\n" + "=" * 80)
print("TABLEAU RECAPITULATIF DES PERFORMANCES SUR LE TEST")
print("=" * 80)

def inverse(y_scaled):
    """Dénormalise les prédictions."""
    dummy = np.zeros((len(y_scaled), dataset.shape[1]))
    dummy[:, 0] = np.ravel(y_scaled)
    return scaler.inverse_transform(dummy)[:, 0]

models = {
    'OLS': linear_model,
    'Ridge (L2)': ridge_model_tuned,
    'Lasso (L1)': lasso_model_tuned,
    'Elastic Net': enet_model_tuned,
    'Random Forest': rf_model,
}
rows = []
for name, mdl in models.items():
    pred = inverse(mdl.predict(x_test_2d))
    rows.append({'Modele': name,
                 'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                 'MAE': mean_absolute_error(y_test, pred),
                 'R2': r2_score(y_test, pred)})
results = pd.DataFrame(rows).set_index('Modele')
print("\n", results.round(4))
results.to_csv('resultats_test.csv')
print("\nRésultats sauvegardés dans 'resultats_test.csv'")

print("\n" + "=" * 80)
print("FIN DE L'EXECUTION")
print("=" * 80)
