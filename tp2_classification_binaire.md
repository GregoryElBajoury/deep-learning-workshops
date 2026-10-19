# TP2 — Répondre par oui ou non : La classification binaire

Dans le TP précédent, nous avons abordé la régression pour prédire des valeurs continues. Dans ce TP, nous entrons dans le domaine de la **classification binaire**. L'objectif n'est plus de deviner un chiffre, mais de prendre une décision tranchée entre deux options (codées $0$ ou $1$) : par exemple, prédire si un e-mail est un spam ou non, ou si une tumeur détectée est maligne ou bénigne. C'est ce second cas d'application clinique que nous allons étudier.

---

## 1. Charger et préparer les données

Nous commençons par charger le jeu de données classique *Breast Cancer* (fourni par `scikit-learn`), qui contient des mesures morphologiques de cellules. 

### Pourquoi inverser la cible ?
Par défaut, ce jeu de données étiquette les tumeurs malignes par `0` et les bénignes par `1`. En Deep Learning, il est de coutume d'attribuer la valeur **`1`** à la **classe d'intérêt** (la condition que l'on cherche à détecter en priorité, comme le risque ou la maladie). C'est pourquoi nous appliquons l'opération `y = 1 - data.target` pour inverser cette logique.

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Chargement du jeu de données brut
data = load_breast_cancer()
X = data.data    # Les caractéristiques (features) décrivant les cellules
y = 1 - data.target  # On redéfinit la cible pour que 1 corresponde à une tumeur maligne

# Affichage des dimensions et de la proportion de la classe positive
print(X.shape, "| part de tumeurs malignes :", round(y.mean(), 3))
```

---

## 2. Découpage et standardisation (Attention au Data Leakage)

Avant d'entraîner le modèle, deux étapes méthodologiques s'imposent :
1. **Le découpage (Train/Test)** : On sépare les données en un ensemble d'entraînement et un ensemble de test. L'argument `stratify=y` garantit que le pourcentage de tumeurs malignes reste identique dans les deux ensembles.
2. **La standardisation** : Les réseaux de neurones réagissent mal aux variables dont les échelles sont trop disparates. Le `StandardScaler` centre et réduit les caractéristiques. 
   > ⚠️ **Règle d'or** : On ajuste le scaler (`fit_transform`) **uniquement sur le jeu d'entraînement**, puis on applique cette transformation au jeu de test (`transform`) pour éviter toute fuite d'information (*Data Leakage*).

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Séparation des données : 80% pour l'apprentissage, 20% pour le test final
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalisation des caractéristiques pour stabiliser l'apprentissage du réseau
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Apprentissage de la moyenne/écart-type sur le train uniquement
X_test = scaler.transform(X_test)       # Application de la même transformation sur le test
```

---

## 3. Les références à battre : Les Baselines

Avant de mobiliser un réseau de neurones, il est indispensable de comparer ses performances à des approches de référence (*baselines*) :
1. **Le modèle naïf** : La performance obtenue en devinant toujours la classe majoritaire.
2. **La Régression Logistique** : Un modèle linéaire simple et robuste qui sert de premier palier de complexité.

```python
from sklearn.linear_model import LogisticRegression

# Calcul de la précision d'un modèle naïf (qui prédit toujours la classe la plus fréquente)
acc_betise = max(y_train.mean(), 1 - y_train.mean())

# Entraînement d'une régression logistique de référence
modele_logistique = LogisticRegression(max_iter=2000)
modele_logistique.fit(X_train, y_train)
acc_logistique = modele_logistique.score(X_test, y_test)

print(f"Précision modèle naïf : {acc_betise:.3f}")
print(f"Précision régression logistique : {acc_logistique:.3f}")
```

---

## 4. Construction et entraînement du réseau de neurones

Pour traiter une tâche de classification binaire avec Keras :
* **La couche de sortie** ne contient qu'**un seul neurone** muni d'une fonction d'activation **sigmoïde** (`activation="sigmoid"`). Celle-ci convertit n'importe quel score brut en une **probabilité** comprise entre $0$ et $1$.
* **La fonction de perte** est la `binary_crossentropy`, mathématiquement adaptée pour pénaliser les erreurs de probabilité sur deux classes.
* **L'arrêt précoce (*Early Stopping*)** surveille la perte sur l'ensemble de validation et stoppe l'entraînement dès que le modèle commence à stagner ou à surapprendre.

```python
from tensorflow import keras
from tensorflow.keras import layers

# Fixation de la graine aléatoire pour assurer la reproductibilité
keras.utils.set_random_seed(42)

# Définition de l'architecture séquentielle du réseau
modele = keras.Sequential([
    layers.Input(shape=(30,)),             # Couche d'entrée correspondant aux 30 caractéristiques
    layers.Dense(16, activation="relu"),   # Première couche cachée (extraction de motifs)
    layers.Dense(8, activation="relu"),    # Deuxième couche cachée (synthèse)
    layers.Dense(1, activation="sigmoid")  # Sortie unique : probabilité que la tumeur soit maligne
])

# Compilation du modèle avec l'optimiseur Adam et la perte binaire
modele.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Configuration de l'arrêt précoce pour stopper l'entraînement au moment opportun
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Lancement de l'entraînement
histoire = modele.fit(
    X_train, y_train,
    epochs=100, batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)
```

---

## 5. Évaluation fine : Matrice de confusion et métriques

L'exactitude globale (*accuracy*) ne suffit pas en milieu médical. Pour analyser finement les erreurs, nous calculons les composantes de la **matrice de confusion** (Vrais Positifs, Vrais Négatifs, Faux Positifs, Faux Négatifs) afin d'en déduire deux métriques majeures :
* **La Précision** : Fiabilité des alertes positives ($VP / [VP + FP]$).
* **Le Rappel** : Capacité du modèle à ne rater aucun cas positif ($VP / [VP + FN]$).

```python
import numpy as np

# Récupération des probabilités prédites puis conversion en classes (seuil par défaut à 0.5)
proba = modele.predict(X_test, verbose=0).ravel()
pred = (proba >= 0.5).astype(int)

# Calcul manuel des différents cas de la matrice de confusion
VP = np.sum((pred == 1) & (y_test == 1))
VN = np.sum((pred == 0) & (y_test == 0))
FP = np.sum((pred == 1) & (y_test == 0))
FN = np.sum((pred == 0) & (y_test == 1))

# Calcul des indicateurs dérivés
precision = VP / (VP + FP)
rappel = VP / (VP + FN)

print(f"VP: {VP}, VN: {VN}, FP: {FP}, FN: {FN}")
print(f"Précision : {precision:.3f} | Rappel : {rappel:.3f}")
```

> 💡 *Réflexion métier* : Un **faux négatif** (laisser passer une tumeur maligne) est l'erreur critique absolue en oncologie, tandis qu'un faux positif engendre une simple alerte à vérifier. Le choix des métriques aide à arbitrer ces risques.

---

## 6. Pilotage du seuil de décision et Courbe ROC

Le seuil de décision par défaut ($0.5$) n'est pas figé : le modifier permet d'ajuster le compromis entre précision et rappel. La **courbe ROC** et l'indicateur **AUC** résument graphiquement la performance du modèle à travers tous les seuils de décision possibles.

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Calcul de l'aire sous la courbe (AUC) et des coordonnées de la courbe ROC
auc = roc_auc_score(y_test, proba)
fpr, tpr, _ = roc_curve(y_test, proba)

# Tracé de la courbe ROC
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"Modèle (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Hasard")
plt.xlabel("Taux de fausses alertes (FPR)")
plt.ylabel("Rappel / Taux de vrais positifs (TPR)")
plt.title("Courbe ROC")
plt.legend(loc="lower right")
plt.show()
```

---

## Ce qu'il faut retenir
* **Architecture** : Une couche de sortie à un seul neurone avec une activation `sigmoid` et une perte `binary_crossentropy`.
* **Évaluation** : Toujours dépasser le stade de l'accuracy en étudiant la matrice de confusion, la précision et le rappel.
* **Flexibilité** : Le seuil de décision et la courbe ROC permettent d'adapter le comportement du modèle aux contraintes concrètes du cas d'usage.