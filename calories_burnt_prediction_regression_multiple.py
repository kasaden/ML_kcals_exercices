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



""" ----------------------- DATA VISUALIZATION (HISTOGRAMMES) ----------------------- """

# 1. AGE : Montre si les participants sont plutôt jeunes, vieux, ou mixtes
sns.histplot(calories_data['Age'], kde=True)
plt.title('Distribution des Ages')
plt.show()

# 2. HEIGHT : Montre si la majorité mesure 160cm, 180cm, etc.
sns.histplot(calories_data['Height'], kde=True)
plt.title('Distribution des Tailles')
plt.show()

# 3. WEIGHT : Montre si les gens pèsent plutôt 60kg, 80kg, 100kg
sns.histplot(calories_data['Weight'], kde=True)
plt.title('Distribution des Poids')
plt.show()



""" ----------------------- RÉGRESSION LINÉAIRE MULTIPLE ----------------------- """

# ÉTAPE 1 : Convertir le genre (texte) en nombre (0 = homme, 1 = femme)
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# ÉTAPE 2 : Séparer les FEATURES (X) et la TARGET (Y)
# X = toutes les variables SAUF User_ID et Calories
# Y = ce qu'on veut prédire (Calories)
X = calories_data.drop(columns=['User_ID', 'Calories'])
Y = calories_data['Calories']

print(f"📊 Nombre de features (variables) : {X.shape[1]}")
print(f"📊 Nombre d'exemples : {X.shape[0]}\n")

# ÉTAPE 3 : Diviser les données en 70% TRAIN et 30% TEST
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=23)

print(f"✅ Données d'entraînement : {X_train.shape[0]} exemples")
print(f"✅ Données de test : {X_test.shape[0]} exemples\n")

# ÉTAPE 4 : Créer et entraîner le modèle de RÉGRESSION LINÉAIRE MULTIPLE
# "Multiple" car on utilise PLUSIEURS variables (Age, Poids, Taille, Durée, etc.)
model = LinearRegression()
model.fit(X_train, Y_train)

print("🎯 Modèle de régression linéaire entraîné !\n")

# ÉTAPE 5 : Afficher les COEFFICIENTS (importance de chaque variable)
coefficients = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print("📈 Coefficients du modèle (impact de chaque variable) :")
print(coefficients)
print(f"\n📍 Intercept (constante) : {model.intercept_:.2f}\n")

# ÉTAPE 6 : Faire des prédictions sur les données de TEST
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

# ÉTAPE 9 : Tester le modèle avec UN exemple concret
# Exemple : Femme (1), 25 ans, 165cm, 60kg, 30min, 120 bpm, 37°C
exemple = pd.DataFrame([[1, 25, 165, 60, 30, 120, 37]], 
                       columns=X.columns)
calories_predites = model.predict(exemple)

print(f"🔥 EXEMPLE DE PRÉDICTION :")
print(f"   Profil : Femme, 25 ans, 165cm, 60kg, 30min d'exercice, 120 bpm, 37°C")
print(f"   Calories prédites : {calories_predites[0]:.2f} calories\n")



""" ----------------------- FORMULES ----------------------- """

# Afficher la formule mathématique du modèle
print("📐 FORMULE MATHÉMATIQUE :")
print(f"   Calories = {model.intercept_:.2f}")
for var, coef in zip(X.columns, model.coef_):
    print(f"            + ({coef:.2f} × {var})")