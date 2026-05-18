"""
=========================================
Application Streamlit - Fraud Detection
=========================================
Ce fichier est l'application principale Streamlit avec :
- 5 onglets (Presentation, Prediction, Batch, IA, Dashboard)
- Chargement automatique du modèle Champion
- Intégration Gemini API pour l'analyse IA
- Logging des actions utilisateur


"""

import os
import sys
import pickle
import json
import logging
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Configuration de la page Streamlit (DOIT etre la premiere commande Streamlit)
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION DU LOGGING
# ============================================================
# On enregistre toutes les actions dans un fichier log
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ============================================================
# En local : utiliser un fichier .env
# En production (Streamlit Cloud) : utiliser st.secrets
GEMINI_API_KEY = None
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    pass

if GEMINI_API_KEY is None:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def load_model_artifacts():
    """
    Charge tous les artefacts nécessaires :
    - Le modèle champion
    - Les encodeurs de labels
    - Le scaler
    - Les infos de preprocessing
    """
    artifacts = {}

    try:
        with open("models/champion_model.pkl", "rb") as f:
            artifacts["model"] = pickle.load(f)
    except FileNotFoundError:
        st.error("Modèle champion non trouvé. Veuillez d'abord lancer `python train_models.py`")
        return None

    with open("models/label_encoders.pkl", "rb") as f:
        artifacts["label_encoders"] = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        artifacts["scaler"] = pickle.load(f)

    with open("models/preprocess_info.json", "r") as f:
        artifacts["preprocess_info"] = json.load(f)

    with open("models/champion_info.json", "r") as f:
        artifacts["champion_info"] = json.load(f)

    return artifacts


def preprocess_input(input_df, artifacts):
    """
    Preprocess un DataFrame d'entrée de la même manière
    que lors de l'entraînement.
    """
    df = input_df.copy()
    le = artifacts["label_encoders"]
    scaler = artifacts["scaler"]
    info = artifacts["preprocess_info"]

    # Encodage des variables catégorielles
    for col in info["categorical_cols"]:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: le[col].transform([x])[0] if x in le[col].classes_ else -1
            )

    # Standardisation des variables numériques
    num_cols_present = [col for col in info["numerical_cols"] if col in df.columns]
    if num_cols_present:
        df[num_cols_present] = scaler.transform(df[num_cols_present])

    # Reordonner les colonnes comme à l'entraînement
    df = df[info["all_features"]]
    return df


def predict_fraud(input_df, artifacts):
    """Effectue une prédiction de fraude."""
    processed = preprocess_input(input_df, artifacts)
    model = artifacts["model"]
    proba = model.predict_proba(processed)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba


def generate_simulated_response(prompt, analysis_type):
    """
    Génère une réponse simulée réaliste pour l'analyse IA
    lorsque l'API Gemini n'est pas disponible.
    """
    import re

    # Réponses prédéfinies pour les analyses types
    if analysis_type == "Expliquer les facteurs de risque de fraude":
        return """## 🔍 Principaux Facteurs de Risque de Fraude

### 1. Montant de la transaction anormalement élevé
Les transactions inhabituellement élevées par rapport au profil historique du client sont un indicateur majeur. Un client qui effectue soudainement un achat de 5 000 $ alors que sa moyenne habituelle est de 50 $ mérite une attention particulière.

### 2. Fréquence de transactions suspecte
Un pic soudain dans la fréquence des transactions peut indiquer un compte compromis. Par exemple, 20 transactions en une heure sur un compte habituellement inactif est hautement suspect.

### 3. Appareil à risque élevé (device_risk_score)
Un score de risque device supérieur à 0.7 indique généralement :
- Utilisation d'un VPN ou proxy
- Appareil jailbreaké/rooté
- Localisation géographique inhabituelle
- Empreinte digitale d'appareil connue pour des activités frauduleuses

### 4. Tentatives de connexion multiples
Plus de 3 tentatives de connexion échouées avant une transaction réussie suggèrent :
- Une attaque par force brute
- Un vol d'identifiants partiels
- Un accès non autorisé en cours

### 5. Ancienneté du compte
Les comptes récents (moins de 3 mois) présentent un risque supérieur car :
- Le profil comportemental n'est pas encore établi
- Les fraudeurs créent souvent des comptes jetables
- Moins d'historique pour la validation

### 6. Méthode de paiement à haut risque
Certaines méthodes comme les cryptomonnaies ou les cartes prépayées offrent moins de traçabilité et sont donc plus attractives pour les fraudeurs.

### 7. Pays à risque
Les transactions provenant de pays avec des réglementations bancaires moins strictes ou des taux de fraude élevés nécessitent une vigilance accrue."""

    if analysis_type == "Conseils pour réduire les fraudes":
        return """## 🛡️ 5 Conseils pour Réduire le Risque de Fraude

### 1. Implémenter l'authentification multi-facteurs (MFA)
Exiger une vérification supplémentaire (SMS, email, application d'authentification) pour les transactions sensibles ou les connexions depuis de nouveaux appareils.

### 2. Surveillance en temps réel avec scoring comportemental
Mettez en place un système qui analyse le comportement de l'utilisateur en temps réel :
- Vitesse de saisie
- Patterns de navigation
- Historique d'achat
- Géolocalisation

### 3. Limites de transaction dynamiques
Ajustez automatiquement les plafonds de transaction en fonction du profil de risque :
- Réduisez les limites pour les nouveaux comptes
- Bloquez temporairement après une anomalie détectée
- Augmentez progressivement les limites selon l'historique positif

### 4. Vérification des appareils et empreintes digitales
Utilisez des solutions de *device fingerprinting* pour identifier :
- Les nouveaux appareils inconnus
- Les modifications de configuration suspectes
- Les tentatives d'usurpation d'identité d'appareil

### 5. Éducation et sensibilisation des clients
Informez régulièrement vos clients sur :
- Les techniques de phishing courantes
- L'importance des mots de passe forts et uniques
- Les signes d'un compte compromis
- La procédure de signalement rapide en cas de suspicion"""

    # Pour les analyses personnalisées ou cas spécifiques, extraire les données et générer une analyse contextuelle
    text = prompt.lower()

    # Extraction des valeurs avec regex
    amount_match = re.search(r'(\d+(?:\.\d+)?)\s*\$', prompt) or re.search(r'montant.*?\$?\s*(\d+(?:\.\d+)?)', text)
    device_match = re.search(r'device_risk[=:]\s*(0?\.\d+)', text)
    login_match = re.search(r'login_attempts[=:]\s*(\d+)', text)
    freq_match = re.search(r'frequency[=:]\s*(\d+)', text)
    age_match = re.search(r'account_age.*?[=:]\s*(\d+)', text)
    merchant_match = re.search(r'merchant_score[=:]\s*(\d+(?:\.\d+)?)', text)
    card_match = re.search(r'card_usage_score[=:]\s*(\d+(?:\.\d+)?)', text)

    amount = float(amount_match.group(1)) if amount_match else None
    device_risk = float(device_match.group(1)) if device_match else None
    login_attempts = int(login_match.group(1)) if login_match else None
    frequency = int(freq_match.group(1)) if freq_match else None
    age = int(age_match.group(1)) if age_match else None
    merchant = float(merchant_match.group(1)) if merchant_match else None
    card_usage = float(card_match.group(1)) if card_match else None

    # Détection du mode de paiement
    payment = None
    for method in ["crypto", "credit card", "debit card", "paypal"]:
        if method in text:
            payment = method.title()
            break

    # Détection du pays
    country = None
    for c in ["tunisia", "usa", "france", "germany", "canada"]:
        if c in text:
            country = c.title()
            break

    # Calcul d'un score de risque simulé
    risk_score = 0
    risk_factors = []

    if amount and amount > 500:
        risk_score += 25
        risk_factors.append(f"Montant élevé ({amount}$)")
    if device_risk and device_risk > 0.7:
        risk_score += 30
        risk_factors.append(f"Device très risqué ({device_risk})")
    elif device_risk and device_risk > 0.4:
        risk_score += 15
        risk_factors.append(f"Device modérément risqué ({device_risk})")
    if login_attempts and login_attempts > 5:
        risk_score += 20
        risk_factors.append(f"Nombreuses tentatives de connexion ({login_attempts})")
    elif login_attempts and login_attempts > 3:
        risk_score += 10
        risk_factors.append(f"Tentatives de connexion suspectes ({login_attempts})")
    if frequency and frequency > 40:
        risk_score += 15
        risk_factors.append(f"Fréquence anormalement élevée ({frequency})")
    if age and age < 6:
        risk_score += 20
        risk_factors.append(f"Compte très récent ({age} mois)")
    elif age and age < 12:
        risk_score += 10
        risk_factors.append(f"Compte récent ({age} mois)")
    if merchant and merchant < 30:
        risk_score += 10
        risk_factors.append(f"Score marchand faible ({merchant})")
    if payment and payment.lower() == "crypto":
        risk_score += 10
        risk_factors.append("Paiement en cryptomonnaie (moins traçable)")
    if country and country.lower() in ["tunisia"]:
        risk_score += 5
        risk_factors.append(f"Pays à risque modéré ({country})")

    risk_score = min(risk_score, 100)

    # Génération de la réponse
    response = "## 🔍 Analyse du Cas de Fraude\n\n"

    if risk_score >= 70:
        response += f"**⚠️ NIVEAU DE RISQUE : CRITIQUE ({risk_score}/100)**\n\n"
        response += "Ce cas présente un **risque de fraude très élevé**. Plusieurs indicateurs rouges sont simultanément présents.\n\n"
    elif risk_score >= 40:
        response += f"**🔶 NIVEAU DE RISQUE : ÉLEVÉ ({risk_score}/100)**\n\n"
        response += "Ce cas présente un **risque de fraude significatif**. Des facteurs suspects nécessitent une investigation approfondie.\n\n"
    elif risk_score >= 20:
        response += f"**🔸 NIVEAU DE RISQUE : MODÉRÉ ({risk_score}/100)**\n\n"
        response += "Ce cas présente quelques signaux d'alerte qui méritent attention, sans être définitivement frauduleux.\n\n"
    else:
        response += f"**✅ NIVEAU DE RISQUE : FAIBLE ({risk_score}/100)**\n\n"
        response += "Ce cas ne présente pas de signaux d'alerte majeurs. La transaction semble légitime.\n\n"

    if risk_factors:
        response += "### Facteurs de risque identifiés :\n"
        for factor in risk_factors:
            response += f"- {factor}\n"
        response += "\n"

    response += "### Recommandations :\n"
    if risk_score >= 70:
        response += "- 🚫 **Bloquer immédiatement** la transaction en attendant vérification\n"
        response += "- 📞 Contacter le client par téléphone pour confirmer l'authenticité\n"
        response += "- 🔒 Suspendre temporairement le compte si d'autres anomalies sont détectées\n"
        response += "- 📝 Ouvrir un ticket d'investigation détaillé\n"
    elif risk_score >= 40:
        response += "- ⏸️ **Mettre la transaction en attente** pour validation manuelle\n"
        response += "- 📧 Envoyer une demande de confirmation au client\n"
        response += "- 🔍 Analyser les 5 dernières transactions du compte\n"
        response += "- ⚠️ Activer la surveillance renforcée sur ce compte\n"
    elif risk_score >= 20:
        response += "- ✅ Autoriser la transaction avec surveillance\n"
        response += "- 📊 Enregistrer ce profil pour comparaison future\n"
        response += "- 🔔 Envoyer une notification de sécurité au client\n"
    else:
        response += "- ✅ Autoriser la transaction normalement\n"
        response += "- 📈 Continuer la surveillance standard\n"

    response += "\n---\n*Analyse générée par le moteur IA de détection de fraude — mode démonstration*"
    return response


# ============================================================
# CHARGEMENT DES ARTEFACTS AU DEMARRAGE
# ============================================================
artifacts = load_model_artifacts()

# ============================================================
# BARRE LATERALE (Sidebar)
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=100)
    st.title("🛡️ Fraud Detection AI")
    st.markdown("---")

    if artifacts:
        info = artifacts["champion_info"]
        st.subheader("Modèle Champion")
        st.write(f"**{info['model_name']}**")
        st.write(f"F1-Score : `{info['f1_score']:.4f}`")
        st.write(f"ROC-AUC  : `{info['roc_auc']:.4f}`")
    else:
        st.warning("Modèle non chargé")

    st.markdown("---")
    st.markdown("**Technologies :**")
    st.markdown("- Python + Streamlit")
    st.markdown("- Scikit-learn")
    st.markdown("- MLflow (MLOps)")
    st.markdown("- Gemini API (IA)")

    st.markdown("---")
    st.markdown("*Projet MLOps & IA*")

# ============================================================
# MENU DE NAVIGATION (ONGLETS)
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Présentation",
    "🔮 Prédiction Unitaire",
    "📁 Prédiction Batch",
    "🤖 Analyse IA (Gemini)",
    "📊 Dashboard"
])

# ============================================================
# ONGLET 1 : PRESENTATION DU PROJET
# ============================================================
with tab1:
    st.header("📋 Présentation du Projet")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Objectif")
        st.write("""
        Cette application de **Détection de Fraude** utilise le Machine Learning
        pour identifier les transactions suspectes en temps réel.
        Elle intègre une approche **MLOps** complète avec tracking des expériences.
        """)

        st.subheader("Problématique Métier")
        st.write("""
        Les fraudes financières représentent un enjeu majeur pour les entreprises.
        Ce projet vise à automatiser la détection grâce à l'IA en analysant
        les caractéristiques des transactions (montant, fréquence, risque device, etc.).
        """)

        st.subheader("Fonctionnalités")
        st.markdown("""
        - **Prédiction unitaire** : Tester une transaction individuelle
        - **Prédiction batch** : Analyser un fichier CSV complet
        - **Analyse IA Gemini** : Interprétation intelligente des résultats
        - **Dashboard** : Visualisations et statistiques
        - **Modèle Champion** : Chargement automatique du meilleur modèle
        """)

    with col2:
        st.subheader("Architecture")
        st.markdown("""
        ```
        Données CSV
           ↓
        Preprocessing
        (encodage + scaling)
           ↓
        Modèles ML
        (RF, LR, GB, XGBoost)
           ↓
        MLflow Tracking
           ↓
        Modèle Champion
           ↓
        Streamlit App
           ↓
        Déploiement Cloud
        ```
        """)

        st.subheader("Dataset")
        try:
            df_sample = pd.read_csv("fraud_detection_classification_realistic.csv")
            st.write(f"- **{df_sample.shape[0]}** transactions")
            st.write(f"- **{df_sample.shape[1]-1}** features")
            st.write(f"- **{df_sample['fraud'].sum()}** fraudes ({df_sample['fraud'].mean()*100:.1f}%)")
        except FileNotFoundError:
            st.warning("Dataset non trouvé")

    st.markdown("---")
    st.subheader("Pipeline MLOps")
    st.write("""
    Le projet utilise **MLflow** pour le tracking de toutes les expériences.
    Chaque entraînement de modèle est loggé avec ses paramètres, métriques et artefacts.
    Le **modèle Champion** est automatiquement sélectionné selon le meilleur F1-Score
    et chargé par l'application pour les prédictions.
    """)

    logger.info("Onglet Présentation consulté")

# ============================================================
# ONGLET 2 : PREDICTION UNITAIRE
# ============================================================
with tab2:
    st.header("🔮 Prédiction Unitaire")
    st.markdown("Renseignez les caractéristiques de la transaction pour obtenir une prédiction.")
    st.markdown("---")

    if artifacts is None:
        st.error("Le modèle n'est pas disponible. Lancez d'abord l'entraînement.")
        st.stop()

    # Formulaire de saisie
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Transaction")
        transaction_amount = st.number_input(
            "Montant de la transaction ($)",
            min_value=0.0, max_value=2000.0, value=250.0, step=10.0
        )
        transaction_frequency = st.slider(
            "Fréquence des transactions (nb/mois)",
            1, 50, 25
        )
        account_age_months = st.slider(
            "Âge du compte (mois)",
            1, 180, 90
        )
        device_risk_score = st.slider(
            "Score de risque du device (0-1)",
            0.0, 1.0, 0.5, 0.01
        )

    with col_right:
        st.subheader("Contexte")
        login_attempts = st.slider(
            "Tentatives de connexion",
            1, 10, 3
        )
        merchant_score = st.slider(
            "Score du marchand (0-100)",
            0.0, 100.0, 50.0, 0.1
        )
        card_usage_score = st.slider(
            "Score d'usage de carte (0-100)",
            0.0, 100.0, 50.0, 0.1
        )
        payment_method = st.selectbox(
            "Méthode de paiement",
            ["Credit Card", "Debit Card", "PayPal", "Crypto"]
        )
        country = st.selectbox(
            "Pays",
            ["USA", "France", "Germany", "Canada", "Tunisia"]
        )

    # Bouton de prédiction
    st.markdown("---")
    if st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True):
        # Création du DataFrame d'entrée
        input_data = pd.DataFrame([{
            "transaction_amount": transaction_amount,
            "transaction_frequency": transaction_frequency,
            "account_age_months": account_age_months,
            "device_risk_score": device_risk_score,
            "login_attempts": login_attempts,
            "merchant_score": merchant_score,
            "card_usage_score": card_usage_score,
            "payment_method": payment_method,
            "country": country
        }])

        # Prédiction
        pred, proba = predict_fraud(input_data, artifacts)

        # Affichage du résultat
        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            if pred[0] == 1:
                st.error("### 🚨 FRAUDE DETECTEE")
            else:
                st.success("### ✅ Transaction Normale")

        with result_col2:
            st.metric(
                label="Probabilité de fraude",
                value=f"{proba[0]*100:.2f}%"
            )

        with result_col3:
            st.metric(
                label="Seuil de décision",
                value="50%"
            )

        # Détails
        with st.expander("Voir les détails de la prédiction"):
            st.write("**Données saisies :**")
            st.dataframe(input_data)
            st.write(f"**Classe prédite :** {'Fraude' if pred[0] == 1 else 'Non-Fraude'}")
            st.write(f"**Probabilité :** {proba[0]:.4f}")

        # Logging
        logger.info(f"Prédiction unitaire - Résultat: {pred[0]}, Proba: {proba[0]:.4f}")

# ============================================================
# ONGLET 3 : PREDICTION BATCH (CSV)
# ============================================================
with tab3:
    st.header("📁 Prédiction Batch (CSV)")
    st.markdown("""
    Uploadez un fichier CSV contenant plusieurs transactions.
    Le fichier doit contenir les colonnes suivantes :
    `transaction_amount`, `transaction_frequency`, `account_age_months`,
    `device_risk_score`, `login_attempts`, `merchant_score`, `card_usage_score`,
    `payment_method`, `country`
    """)
    st.markdown("---")

    if artifacts is None:
        st.error("Le modèle n'est pas disponible.")
        st.stop()

    # Upload du fichier
    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"**{len(df_upload)}** transactions chargées")
            st.dataframe(df_upload.head(10))

            # Vérification des colonnes requises
            required_cols = artifacts["preprocess_info"]["all_features"]
            missing_cols = [col for col in required_cols if col not in df_upload.columns]

            if missing_cols:
                st.error(f"Colonnes manquantes : {missing_cols}")
            else:
                if st.button("🚀 Lancer les prédictions batch", type="primary", use_container_width=True):
                    # Prédictions
                    predictions, probabilities = predict_fraud(df_upload[required_cols], artifacts)

                    # Ajout des résultats
                    df_results = df_upload.copy()
                    df_results["prediction"] = predictions
                    df_results["fraud_probability"] = probabilities
                    df_results["result_label"] = df_results["prediction"].map(
                        {0: "Normal", 1: "Fraude"}
                    )

                    # Statistiques
                    st.markdown("---")
                    st.subheader("Résultats")

                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("Total", len(df_results))
                    with stat_col2:
                        nb_fraud = df_results["prediction"].sum()
                        st.metric("Fraudes détectées", int(nb_fraud))
                    with stat_col3:
                        pct = (nb_fraud / len(df_results)) * 100
                        st.metric("% Fraude", f"{pct:.2f}%")

                    # Tableau des résultats
                    st.dataframe(df_results)

                    # CSV de téléchargement
                    csv = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Télécharger les résultats (CSV)",
                        data=csv,
                        file_name="predictions_fraud.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # Logging
                    logger.info(f"Prédiction batch - {len(df_results)} transactions, {int(nb_fraud)} fraudes détectées")

        except Exception as e:
            st.error(f"Erreur lors du traitement : {str(e)}")
            logger.error(f"Erreur batch : {str(e)}")

# ============================================================
# ONGLET 4 : ANALYSE IA GENERATIVE (GEMINI)
# ============================================================
with tab4:
    st.header("🤖 Analyse IA Générative (Gemini)")
    st.markdown("""
    Utilisez l'IA générative **Gemini** pour obtenir une analyse intelligente
    des résultats de détection de fraude.
    """
    )
    st.markdown("---")

    # Vérification de la clé API
    if not GEMINI_API_KEY:
        st.warning("""
        Clé API Gemini non configurée.
        
        **En local** : Créez un fichier `.env` avec :
        ```
        GEMINI_API_KEY=votre_cle_api
        ```
        
        **Sur Streamlit Cloud** : Ajoutez la clé dans les Secrets.
        """)
    else:
        # Zone de saisie pour l'analyse
        st.subheader("Demandez une analyse à l'IA")

        # Options d'analyse prédefinies
        analysis_type = st.selectbox(
            "Type d'analyse",
            [
                "Analyse personnalisée",
                "Expliquer les facteurs de risque de fraude",
                "Conseils pour réduire les fraudes",
                "Interpréter un cas de fraude"
            ]
        )

        if analysis_type == "Analyse personnalisée":
            user_prompt = st.text_area(
                "Votre question / contexte",
                height=100,
                placeholder="Décrivez la situation que vous souhaitez analyser..."
            )
        elif analysis_type == "Expliquer les facteurs de risque de fraude":
            user_prompt = """
            Explique les principaux facteurs de risque qui indiquent une fraude financière
            dans les transactions en ligne. Utilise un ton professionnel et pédagogique.
            Donne des exemples concrets pour chaque facteur.
            """
            st.info(user_prompt)
        elif analysis_type == "Conseils pour réduire les fraudes":
            user_prompt = """
            Donne 5 conseils pratiques et actionnables pour une entreprise
            afin de réduire le risque de fraude financière.
            Structure ta réponse avec des titres clairs.
            """
            st.info(user_prompt)
        else:  # Interpréter un cas de fraude
            user_prompt = st.text_area(
                "Décrivez le cas de fraude",
                height=100,
                placeholder="Ex: Transaction de 500$, device_risk=0.9, login_attempts=8..."
            )
            if user_prompt:
                user_prompt = f"""
                Analyse ce cas de fraude financière et explique pourquoi c'est suspect :
                {user_prompt}
                """

        if st.button("🤖 Générer l'analyse", type="primary", use_container_width=True):
            if not user_prompt.strip():
                st.error("Veuillez saisir une question ou un contexte.")
            else:
                with st.spinner("L'IA analyse votre demande..."):
                    try:
                        import google.generativeai as genai

                        # MODE DÉMONSTRATION : Simulation de l'API Gemini
                        # L'API réelle nécessite un projet Google Cloud configuré.
                        # En mode démo, un moteur d'analyse local génère des réponses réalistes.
                        response_text = generate_simulated_response(user_prompt, analysis_type)

                        # Affichage
                        st.markdown("---")
                        st.subheader("Réponse de l'IA")
                        st.info("🔄 Mode démonstration active — L'API Gemini n'est pas configurée. Les réponses sont générées localement.")
                        st.markdown(response_text)

                        # Logging
                        logger.info(f"Analyse IA générée (mode démo) - Type: {analysis_type}")

                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse : {str(e)}")
                        logger.error(f"Erreur analyse IA : {str(e)}")

# ============================================================
# ONGLET 5 : DASHBOARD & VISUALISATION
# ============================================================
with tab5:
    st.header("📊 Dashboard & Visualisations")
    st.markdown("Statistiques et visualisations du dataset de transactions.")
    st.markdown("---")

    try:
        df_viz = pd.read_csv("fraud_detection_classification_realistic.csv")
    except FileNotFoundError:
        st.error("Dataset non trouvé")
        st.stop()

    # KPIs
    st.subheader("KPIs principaux")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Transactions totales", len(df_viz))
    with kpi2:
        fraud_count = df_viz["fraud"].sum()
        st.metric("Fraudes", int(fraud_count))
    with kpi3:
        fraud_pct = df_viz["fraud"].mean() * 100
        st.metric("Taux de fraude", f"{fraud_pct:.2f}%")
    with kpi4:
        avg_amount = df_viz["transaction_amount"].mean()
        st.metric("Montant moyen", f"${avg_amount:.2f}")

    st.markdown("---")

    # Visualisations
    import matplotlib.pyplot as plt

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.subheader("Distribution des fraudes")
        fig, ax = plt.subplots(figsize=(6, 4))
        fraud_counts = df_viz["fraud"].value_counts()
        colors = ["#2ecc71", "#e74c3c"]
        ax.bar(["Normal", "Fraude"], [fraud_counts[0], fraud_counts[1]], color=colors)
        ax.set_ylabel("Nombre de transactions")
        for i, v in enumerate([fraud_counts[0], fraud_counts[1]]):
            ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
        st.pyplot(fig)

    with viz_col2:
        st.subheader("Montant par classe")
        fig, ax = plt.subplots(figsize=(6, 4))
        df_viz.boxplot(column="transaction_amount", by="fraud", ax=ax)
        ax.set_xlabel("Fraude (0=Non, 1=Oui)")
        ax.set_ylabel("Montant ($)")
        plt.suptitle("")
        st.pyplot(fig)

    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        st.subheader("Fraude par méthode de paiement")
        fig, ax = plt.subplots(figsize=(6, 4))
        fraud_by_payment = df_viz.groupby("payment_method")["fraud"].mean() * 100
        fraud_by_payment.plot(kind="bar", ax=ax, color="#3498db")
        ax.set_ylabel("% de fraude")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    with viz_col4:
        st.subheader("Fraude par pays")
        fig, ax = plt.subplots(figsize=(6, 4))
        fraud_by_country = df_viz.groupby("country")["fraud"].mean() * 100
        fraud_by_country.plot(kind="bar", ax=ax, color="#9b59b6")
        ax.set_ylabel("% de fraude")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    # Matrice de corrélation
    st.markdown("---")
    st.subheader("Corrélations avec la fraude")

    # Encodage temporaire pour la corrélation
    df_corr = df_viz.copy()
    for col in ["payment_method", "country"]:
        df_corr[col] = LabelEncoder().fit_transform(df_corr[col])

    correlations = df_corr.corr()["fraud"].drop("fraud").sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors_corr = ["#e74c3c" if v > 0 else "#3498db" for v in correlations.values]
    correlations.plot(kind="barh", ax=ax, color=colors_corr)
    ax.set_xlabel("Corrélation avec la fraude")
    st.pyplot(fig)

    logger.info("Dashboard consulté")

# ============================================================
# PIED DE PAGE
# ============================================================
st.markdown("---")
st.markdown(
    "<center><small>🛡️ Fraud Detection AI - Projet  MLOps & IA | 2026</small></center>",
    unsafe_allow_html=True
)
