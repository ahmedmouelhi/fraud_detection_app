"""
=========================================
Script d'entraînement ML - Fraud Detection
=========================================
Ce script :
1. Charge et prépare les données
2. Entraîne 4 modèles de classification
3. Log les expériences dans MLflow
4. Sauvegarde le modèle Champion

Pour l'examen, ce script montre :
- Le prétraitement des données (nettoyage, encodage, split)
- L'entraînement de plusieurs modèles ML
- Le tracking MLOps avec MLflow
- La sélection du meilleur modèle (Champion)
"""

import os
import warnings
import pickle
import json

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# MLflow pour le tracking MLOps
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "fraud_detection_classification_realistic.csv"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")  # DagsHub si .env configuré, sinon local
EXPERIMENT_NAME = "Fraud_Detection_Experiment"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Créer le dossier pour le modèle champion
os.makedirs("models", exist_ok=True)

# ============================================================
# 1. CHARGEMENT ET EXPLORATION DES DONNEES
# ============================================================
print("=" * 50)
print("1. CHARGEMENT DES DONNEES")
print("=" * 50)

df = pd.read_csv(DATA_PATH)
print(f"Dimensions du dataset : {df.shape}")
print(f"Colonnes : {list(df.columns)}")
print(f"\nDistribution de la target (fraud) :")
print(df["fraud"].value_counts())
print(f"\nPourcentage de fraude : {df['fraud'].mean()*100:.2f}%")

# ============================================================
# 2. PREPROCESSING DES DONNEES
# ============================================================
print("\n" + "=" * 50)
print("2. PREPROCESSING")
print("=" * 50)

# Séparation features / target
X = df.drop("fraud", axis=1)
y = df["fraud"]

# Identifier les colonnes catégorielles et numériques
categorical_cols = ["payment_method", "country"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

print(f"Colonnes numériques : {numerical_cols}")
print(f"Colonnes catégorielles : {categorical_cols}")

# Encodage des variables catégorielles avec LabelEncoder
# On sauvegarde les encodeurs pour les réutiliser dans l'app
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"Encodage {col} : {list(le.classes_)}")

# Sauvegarder les encodeurs
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Standardisation des features numériques
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Sauvegarder le scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Sauvegarder les informations des colonnes
preprocess_info = {
    "numerical_cols": numerical_cols,
    "categorical_cols": categorical_cols,
    "all_features": list(X.columns)
}
with open("models/preprocess_info.json", "w") as f:
    json.dump(preprocess_info, f, indent=2)

print(f"\nFeatures après preprocessing : {list(X.columns)}")

# ============================================================
# 3. SPLIT TRAIN / TEST
# ============================================================
print("\n" + "=" * 50)
print("3. SPLIT TRAIN / TEST")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train set : {X_train.shape[0]} échantillons")
print(f"Test set  : {X_test.shape[0]} échantillons")
print(f"Fraude dans train : {y_train.mean()*100:.2f}%")
print(f"Fraude dans test  : {y_test.mean()*100:.2f}%")

# ============================================================
# 4. CONFIGURATION MLFLOW
# ============================================================
print("\n" + "=" * 50)
print("4. CONFIGURATION MLFLOW")
print("=" * 50)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Tracking URI : {MLFLOW_TRACKING_URI}")
print(f"Experiment   : {EXPERIMENT_NAME}")

# ============================================================
# 5. DEFINITION DES MODELES
# ============================================================
print("\n" + "=" * 50)
print("5. ENTRAINEMENT DES MODELES")
print("=" * 50)

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"  # Gère le déséquilibre des classes
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE
    ),
}

# Ajouter XGBoost si disponible
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    print("XGBoost disponible et ajouté.")
except ImportError:
    print("XGBoost non installé, ignoré.")

# ============================================================
# 6. ENTRAINEMENT ET TRACKING MLFLOW
# ============================================================
print("\n" + "=" * 50)
print("6. ENTRAINEMENT + TRACKING MLFLOW")
print("=" * 50)

results = []

for model_name, model in models.items():
    print(f"\n--- Entraînement : {model_name} ---")

    with mlflow.start_run(run_name=model_name):
        # Log des paramètres du modèle
        params = model.get_params()
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Entraînement
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Métriques
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_proba)

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc:.4f}")

        # Log des métriques dans MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        # Log du modèle
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Stocker les résultats
        results.append({
            "model_name": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc,
            "model": model,
            "run_id": mlflow.active_run().info.run_id
        })

print("\n" + "=" * 50)
print("7. RESULTATS COMPARATIFS")
print("=" * 50)

results_df = pd.DataFrame(results)
print(results_df[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))

# ============================================================
# 8. SELECTION DU MODELE CHAMPION (meilleur F1-Score)
# ============================================================
print("\n" + "=" * 50)
print("8. MODELE CHAMPION")
print("=" * 50)

champion_idx = results_df["f1_score"].idxmax()
champion = results_df.iloc[champion_idx]

print(f"Champion : {champion['model_name']}")
print(f"F1-Score : {champion['f1_score']:.4f}")
print(f"ROC-AUC  : {champion['roc_auc']:.4f}")
print(f"Run ID   : {champion['run_id']}")

# Sauvegarder le modèle champion avec pickle
champion_model = champion["model"]
with open("models/champion_model.pkl", "wb") as f:
    pickle.dump(champion_model, f)

# Sauvegarder les infos du champion
champion_info = {
    "model_name": champion["model_name"],
    "f1_score": float(champion["f1_score"]),
    "roc_auc": float(champion["roc_auc"]),
    "run_id": champion["run_id"],
    "metrics": {
        "accuracy": float(champion["accuracy"]),
        "precision": float(champion["precision"]),
        "recall": float(champion["recall"]),
        "f1_score": float(champion["f1_score"]),
        "roc_auc": float(champion["roc_auc"])
    }
}
with open("models/champion_info.json", "w") as f:
    json.dump(champion_info, f, indent=2)

print("\nModèle champion sauvegardé dans models/champion_model.pkl")
print("Infos du champion sauvegardées dans models/champion_info.json")

# ============================================================
# 9. RAPPORT DE CLASSIFICATION DU CHAMPION
# ============================================================
print("\n" + "=" * 50)
print("9. RAPPORT CLASSIFICATION (CHAMPION)")
print("=" * 50)

y_pred_champion = champion_model.predict(X_test)
print(classification_report(y_test, y_pred_champion, target_names=["Non-Fraude", "Fraude"]))

print("\n" + "=" * 50)
print("ENTRAINEMENT TERMINE !")
print("=" * 50)
print("\nProchaines étapes :")
print("1. Lancer MLflow UI : mlflow ui --backend-store-uri ./mlruns")
print("2. Lancer l'app     : streamlit run app.py")
