# 🛡️ Fraud Detection AI - Projet Examen MLOps & IA

Application complète de détection de fraude avec Machine Learning, tracking MLOps et IA générative.

---

## 📋 Structure du Projet

```
fraud_detection_app/
│
├── app.py                          # Application Streamlit (point d'entrée)
├── train_models.py                 # Script d'entraînement ML + MLflow
├── requirements.txt                # Dépendances Python
├── .env.example                    # Exemple de variables d'environnement
├── .gitignore                      # Fichiers ignorés par Git
├── README.md                       # Ce fichier
│
├── fraud_detection_classification_realistic.csv   # Dataset
│
├── models/                         # Modèles sauvegardés
│   ├── champion_model.pkl          # Modèle Champion (meilleur F1-Score)
│   ├── champion_info.json          # Infos du modèle Champion
│   ├── label_encoders.pkl          # Encodeurs de labels
│   ├── scaler.pkl                  # StandardScaler
│   └── preprocess_info.json        # Infos de preprocessing
│
├── mlruns/                         # Tracking MLflow (local)
└── logs/                           # Logs de l'application
```

---

## 🚀 Installation & Lancement

### 1. Cloner le projet

```bash
git clone <votre-repo-github>
cd fraud_detection_app
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditez .env et ajoutez votre clé API Gemini
```

### 5. Entraîner les modèles

```bash
python train_models.py
```

### 6. Lancer l'application Streamlit

```bash
streamlit run app.py
```

---

## 📊 Fonctionnalités de l'Application

| Onglet | Description |
|--------|-------------|
| 📋 Présentation | Vue d'ensemble du projet et architecture |
| 🔮 Prédiction Unitaire | Tester une transaction individuelle |
| 📁 Prédiction Batch | Analyser un fichier CSV complet |
| 🤖 Analyse IA (Gemini) | Analyse intelligente avec IA générative |
| 📊 Dashboard | Visualisations et statistiques |

---

## 🧠 Pipeline Machine Learning

### Modèles entraînés :
- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost** (si installé)

### Métriques trackées :
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Sélection du Champion :
Le modèle avec le meilleur **F1-Score** est automatiquement sélectionné comme Champion.

---

## 🔧 Tracking MLOps avec MLflow

### Lancer l'UI MLflow :
```bash
mlflow ui --backend-store-uri ./mlruns
```
Accès : http://localhost:5000

### Configuration DagsHub (production) :
```python
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/USERNAME/REPO.mlflow")
```

---

## 🤖 Configuration Gemini API

1. Créez une clé sur [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Ajoutez-la dans le fichier `.env` :
```env
GEMINI_API_KEY=votre_cle_ici
```

---

## 🌐 Déploiement

### Streamlit Community Cloud
1. Poussez le code sur GitHub
2. Connectez votre repo sur [share.streamlit.io](https://share.streamlit.io)
3. Ajoutez `GEMINI_API_KEY` dans les Secrets

### Hugging Face Spaces
1. Créez un Space Streamlit sur [huggingface.co](https://huggingface.co)
2. Poussez le code
3. Configurez les secrets

---

## 📝 Points Clés pour l'Examen

### Architecture :
- **Préprocessing** : Encodage + Standardisation
- **MLflow** : Tracking des expériences
- **Champion** : Chargement automatique du meilleur modèle
- **Streamlit** : Interface utilisateur interactive
- **Gemini** : Analyse IA générative

### Technologies utilisées :
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit
- MLflow + DagsHub
- Google Gemini API
- Git/GitHub

---

## 👨‍💻 Auteur

Projet réalisé  **MLOps & IA Générative**.
