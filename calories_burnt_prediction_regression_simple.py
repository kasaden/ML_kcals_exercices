import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

"""Data Collection & Processing"""

# loading the data from csv file to a Pandas DataFrame
calories = pd.read_csv('calories.csv')

# print the first 5 rows of the dataframe
calories.head()

exercise_data = pd.read_csv('exercise.csv')

exercise_data.head()

"""Combining the two Dataframes"""

calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

calories_data.head()

# checking the number of rows and columns
calories_data.shape

# getting some informations about the data
calories_data.info()



""" ----------------------- PREPROCESSING THE DATA ----------------------- """

# Vérifier les valeurs manquantes ou nulles
calories_data.isnull().sum()

# Traiter les valeurs manquantes ou nulles
print("Valeurs manquantes : ")
print(calories_data.isnull().sum())

# Suprimer les lignes des valeurs manquantes ou nulles
calories_data = calories_data.dropna()

# Vérifier et supprimer les doublons
print(f"\nNombre de doublons : {calories_data.duplicated().sum()}")
calories_data = calories_data.drop_duplicates()

# Nombre de lignes restantes après le nettoyage
print(f"\nNombre de lignes après le nettoyage : {calories_data.shape[0]}\n")

# Afficher les statistiques descriptives (aberrantes)
print(calories_data.describe())



""" ----------------------- ANALYSE DE CORRÉLATION ----------------------- """

# Convertir le genre en numérique pour l'analyse de corrélation
calories_data_numeric = calories_data.copy()
calories_data_numeric.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# Calculer la matrice de corrélation
correlation_matrix = calories_data_numeric.corr()

# 1. Corrélation avec la cible (Calories)
print("\n🎯 Corrélation avec Calories :")
print(correlation_matrix['Calories'].drop('Calories').sort_values(ascending=False))

# 2. Vérifier l'indépendance entre variables (multicolinéarité)
print("\n🔍 Multicolinéarité (corrélations entre variables) :")
features_only = correlation_matrix.drop(['User_ID', 'Calories'], axis=0).drop(['User_ID', 'Calories'], axis=1)
for i in range(len(features_only.columns)):
    for j in range(i+1, len(features_only.columns)):
        corr_value = features_only.iloc[i, j]
        if abs(corr_value) > 0.7:
            print(f"   ⚠️  {features_only.columns[i]} ↔ {features_only.columns[j]} : {corr_value:.3f}")

# 3. Visualisation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
plt.title('Matrice de Corrélation')
plt.tight_layout()
plt.show()



""" ----------------------- SÉLECTION DES VARIABLES ----------------------- """

# Analyse des résultats de corrélation :
# - Duration, Heart_Rate, Body_Temp sont TRÈS corrélés avec Calories (0.82 à 0.96) ✅
# - MAIS ils sont aussi très corrélés entre eux (multicolinéarité) ❌
# - Gender, Age, Height, Weight ont une corrélation quasi nulle avec Calories ❌

# Décision : Garder uniquement Duration (meilleure corrélation 0.96) et supprimer les autres
# pour éviter la multicolinéarité tout en gardant une forte prédiction

print("\n🔧 SÉLECTION DES VARIABLES :")
print("Variables supprimées (faible corrélation ou multicolinéarité) :")
print("  - Gender, Age, Height, Weight (corrélation ≈ 0 avec Calories)")
print("  - Heart_Rate, Body_Temp (multicolinéarité avec Duration)")
print("\nVariable gardée :")
print("  - Duration (corrélation = 0.96 avec Calories, pas de multicolinéarité)\n")

# Convertir le genre pour garder une copie des données
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)



""" ----------------------- DATA VISUALIZATION (HISTOGRAMMES) ----------------------- """

# Visualiser la distribution de la variable gardée
sns.histplot(calories_data['Duration'], kde=True)
plt.title('Distribution de la Durée d\'exercice')
plt.xlabel('Durée (minutes)')
plt.show()



""" ----------------------- RÉGRESSION LINÉAIRE (SIMPLE) ----------------------- """

# ÉTAPE 1 : Séparer les FEATURES (X) et la TARGET (Y)
# X = uniquement Duration (variable la plus pertinente)
# Y = ce qu'on veut prédire (Calories)
X = calories_data[['Duration']]  # Garder uniquement Duration
Y = calories_data['Calories']

print(f"📊 Nombre de features (variables) : {X.shape[1]}")
print(f"📊 Nombre d'exemples : {X.shape[0]}\n")

# ÉTAPE 2 : Diviser les données en 70% TRAIN et 30% TEST
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=23)

print(f"✅ Données d'entraînement : {X_train.shape[0]} exemples")
print(f"✅ Données de test : {X_test.shape[0]} exemples\n")

# ÉTAPE 3 : Créer et entraîner le modèle de RÉGRESSION LINÉAIRE SIMPLE
# "Simple" car on utilise UNE SEULE variable (Duration)
model = LinearRegression()
model.fit(X_train, Y_train)

print("🎯 Modèle de régression linéaire entraîné !\n")

# ÉTAPE 4 : Afficher le COEFFICIENT (impact de Duration sur Calories)
print("📈 Coefficient du modèle :")
print(f"   Duration : {model.coef_[0]:.2f} calories/minute")
print(f"   Intercept (constante) : {model.intercept_:.2f}\n")
print(f"   → Formule : Calories = {model.intercept_:.2f} + ({model.coef_[0]:.2f} × Duration)\n")

# ÉTAPE 5 : Faire des prédictions sur les données de TEST
predictions = model.predict(X_test)



""" ----------------------- RÉSULTATS ET PRÉCISION (PERFORMANCE DU MODÈLE)----------------------- """

# MAE = Erreur Absolue Moyenne (en calories)
mae = metrics.mean_absolute_error(Y_test, predictions)
print(f"❌ Erreur Moyenne Absolue (MAE) : {mae:.2f} calories")

# R² Score = Qualité du modèle (entre 0 et 1)
r2 = metrics.r2_score(Y_test, predictions)
print(f"✅ Score R² : {r2:.3f}\n")

# ÉTAPE 8 : Visualiser les prédictions VS réalité
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, predictions, alpha=0.5, color='blue')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('Calories Réelles')
plt.ylabel('Calories Prédites')
plt.title('Régression Linéaire Multiple : Prédictions vs Réalité')
plt.grid(True, alpha=0.3)
plt.show()



""" ----------------------- PRÉDICTIONS ----------------------- """

# ÉTAPE 6 : Tester le modèle avec UN exemple concret
# Exemple : 30 minutes d'exercice
exemple_duration = 30
exemple = pd.DataFrame([[exemple_duration]], columns=['Duration'])
calories_predites = model.predict(exemple)

print(f"🔥 EXEMPLE DE PRÉDICTION :")
print(f"   Durée d'exercice : {exemple_duration} minutes")
print(f"   Calories prédites : {calories_predites[0]:.2f} calories\n")



""" ----------------------- FORMULES ----------------------- """

# Afficher la formule mathématique du modèle
print("📐 FORMULE MATHÉMATIQUE FINALE :")
print(f"   Calories = {model.intercept_:.2f} + ({model.coef_[0]:.2f} × Duration)")
print(f"\n   Exemple : Pour 30 min → Calories = {model.intercept_:.2f} + ({model.coef_[0]:.2f} × 30) = {calories_predites[0]:.2f} cal")