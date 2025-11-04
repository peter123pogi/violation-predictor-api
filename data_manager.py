import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import joblib
from threading import Thread
import os

# =======================================================================
# 🧩 UNIFIED DATA MANAGER (Word2Vec + Keyword Hybrid + Disciplinary Mapping)
# =======================================================================
class DataManager:
    """
    Unified data manager for:
      • Violation Predictor (Logistic + Decision Tree)
      • Complaint Context Analyzer (Word2Vec + Keyword Hybrid)
      • Disciplinary Action Mapping (for Decision Tree outputs)
    """

    def __init__(self, dataset_path: str):
        self.upload_path = './dataset'
        self.dataset_path = dataset_path
        self.data = None
        self.word2vec_model = None
        self.vector_size = 100

        # ----- Violation dataset -----
        self.violation_features = [
            "frequency",
            "repeated_major_count",
            "repeated_minor_count",
            "ongoing_incident_count",
            "total_incidents",
            "is_major",
            "no_violation_count",
            "avg_severity",
            "behavior_ratio"
        ]
        self.target_risk = "will_reoffend_same_violation"
        self.target_recommendation = "recommendation"

        # ----- Complaint dataset -----
        self.text_column = "complaint_text"
        self.target_context = "context_label"

        # ----- Models -----
        self.reoffense_model = None
        self.recommendation_model = None
        self.complaint_model = None

        # ----- Keyword mapping -----
        self.keyword_map = {
            "cyber": "Cyberbullying",
            "harass": "Bullying",
            "threat": "Violence",
            "fight": "Altercation",
            "steal": "Theft",
            "cheat": "Academic Dishonesty",
            "vape": "Smoking",
            "alcohol": "Substance Abuse",
            "disrespect": "Misconduct",
            "tardy": "Attendance",
        }

        # ----- Disciplinary Actions Mapping -----
        self.disciplinary_actions = {
            "1": "Warning",
            "2": "Oral Warning",
            "3": "Written Warning",
            "4": "Conference With Parents/Guardians",
            "5": "Restitution",
            "6": "Confiscation",
            "7": "Demerit",
            "8": "Suspension",
            "9": "Exclusion",
            "10": "Dismissal/Non-readmission",
            "11": "Expulsion"
        }
    def append_dataset_list(self, data):
        file_path = "dataset/dataset-list.csv"
        new_data = pd.DataFrame([data])
        
        new_data.to_csv(file_path, mode='a', header=False, index=False)
        
        
    def get_all_dataset(self):
        df = pd.read_csv('dataset/dataset-list.csv')
        return df
    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------
    def load_dataset(self):
        self.data = pd.read_csv(self.dataset_path)
        print(f"✅ Loaded dataset: {len(self.data)} rows × {len(self.data.columns)} columns")
        return self.data

    # --------------------------------------------------------
    # Detect Dataset Type
    # --------------------------------------------------------
    def detect_dataset_type(self):
        if self.text_column in self.data.columns:
            print("💬 Detected COMPLAINT dataset (Word2Vec + Keyword Hybrid)")
            return "complaint"
        elif set(self.violation_features).issubset(self.data.columns):
            print("⚙️ Detected VIOLATION dataset (numeric-based)")
            return "violation"
        else:
            raise ValueError("❌ Cannot detect dataset type")

    # --------------------------------------------------------
    # ===== VIOLATION PIPELINE =====
    # --------------------------------------------------------
    def clean_violation_data(self):
        initial_len = len(self.data)
        self.data.dropna(inplace=True)
        print(f"🧹 Cleaned violation data: {initial_len - len(self.data)} rows removed.")
        return self.data

    def compute_violation_features(self):
        self.data["avg_severity"] = self.data.apply(self._auto_compute_avg_severity, axis=1)
        self.data["behavior_ratio"] = self.data.apply(self._auto_compute_behavior_ratio, axis=1)
        print("📊 Computed derived features for violation dataset.")
        return self.data

    def _auto_compute_avg_severity(self, row):
        major = row.get("repeated_major_count", 0)
        minor = row.get("repeated_minor_count", 0)
        total = max(1, major + minor)
        return round((major * 1.0 + minor * 0.5) / total, 2)

    def _auto_compute_behavior_ratio(self, row):
        total = max(1, row.get("total_incidents", 0))
        clean = row.get("no_violation_count", 0)
        return round(clean / total, 3)

    def train_violation_models(self):
        """Train Logistic Regression for reoffense and Decision Tree for recommendation."""
        print("🚀 Training Violation Models...")

        X = self.data[self.violation_features]
        y_risk = self.data[self.target_risk]
        y_rec = self.data[self.target_recommendation]

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_risk, test_size=0.2, random_state=42)
        X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X, y_rec, test_size=0.2, random_state=42)

        # Logistic Regression for risk
        self.reoffense_model = LogisticRegression(max_iter=1000)
        self.reoffense_model.fit(X_train_r, y_train_r)
        y_pred_r = self.reoffense_model.predict(X_test_r)
        print("\n🔹 Logistic Regression — Reoffense Risk")
        print(classification_report(y_test_r, y_pred_r))
        print(f"Accuracy: {accuracy_score(y_test_r, y_pred_r):.4f}")

        # Decision Tree for recommendation
        self.recommendation_model = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.recommendation_model.fit(X_train_v, y_train_v)
        y_pred_v = self.recommendation_model.predict(X_test_v)

        # Translate numeric outputs into human-readable disciplinary actions
        y_pred_v_labels = [self.disciplinary_actions.get(str(int(x)), "Unknown") for x in y_pred_v]

        print("\n🌳 Decision Tree — Disciplinary Recommendation Prediction")
        print(classification_report(y_test_v, y_pred_v))
        print(f"Accuracy: {accuracy_score(y_test_v, y_pred_v):.4f}")

        joblib.dump(self.reoffense_model, "reoffense_model.pkl")
        joblib.dump(self.recommendation_model, "recommendation_model.pkl")
        print("💾 Saved Violation Models (reoffense_model.pkl, recommendation_model.pkl)")

    # --------------------------------------------------------
    # ===== COMPLAINT CONTEXT ANALYZER (Word2Vec + Keyword Hybrid) =====
    # --------------------------------------------------------
    def preprocess_complaints(self):
        self.data[self.text_column] = self.data[self.text_column].apply(self._clean_text)
        print("🧽 Cleaned complaint texts.")
        return self.data

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize_texts(self):
        """Tokenize text into lowercase words (no NLTK)."""
        def simple_tokenize(text):
            text = str(text).lower()
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text.split(" ") if text else []
        return [simple_tokenize(t) for t in self.data[self.text_column]]

    def train_word2vec_model(self, tokenized_texts):
        print("🔤 Training Word2Vec model...")
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=5,
            min_count=1,
            workers=4,
            sg=1
        )
        print("✅ Word2Vec model trained.")
        return self.word2vec_model

    def _get_sentence_vector(self, tokens):
        """Compute average vector for a sentence."""
        if not tokens:
            return np.zeros(self.vector_size)
        vectors = [self.word2vec_model.wv[w] for w in tokens if w in self.word2vec_model.wv]
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)

    def add_keyword_feature(self, text):
        """Add keyword context signals."""
        for kw, label in self.keyword_map.items():
            if kw in text:
                return label
        return "General"

    def train_complaint_model(self):
        """Train hybrid complaint context analyzer."""
        print("🚀 Training Complaint Context Analyzer (Hybrid)...")

        tokenized_texts = self._tokenize_texts()
        self.train_word2vec_model(tokenized_texts)
        vectors = np.array([self._get_sentence_vector(toks) for toks in tokenized_texts])

        # Add keyword-based context
        self.data["keyword_label"] = self.data[self.text_column].apply(self.add_keyword_feature)
        X = np.hstack([vectors, pd.get_dummies(self.data["keyword_label"]).values])
        y = self.data[self.target_context]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train Decision Tree classifier
        self.complaint_model = DecisionTreeClassifier(max_depth=12, random_state=42)
        self.complaint_model.fit(X_train, y_train)
        y_pred = self.complaint_model.predict(X_test)

        print("\n💬 Complaint Context Analyzer (Word2Vec + Keyword Hybrid)")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        joblib.dump(self.complaint_model, "complaint_analyzer.pkl")
        joblib.dump(self.word2vec_model, "complaint_word2vec.model")
        print("💾 Saved Complaint Models (complaint_analyzer.pkl, complaint_word2vec.model)")

    # --------------------------------------------------------
    # Save Dataset
    # --------------------------------------------------------
    def save_dataset(self, output_path):
        self.data.to_csv(output_path, index=False)
        print(f"💾 Saved processed dataset to {output_path}")
        

