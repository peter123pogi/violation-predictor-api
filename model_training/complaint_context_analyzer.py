import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re

# -------------------------------------------------------------------
# 🔧 Utility: Convert NumPy → Native Python for JSON serialization
# -------------------------------------------------------------------
def make_json_safe(data):
    """Recursively convert NumPy and pandas types into JSON-serializable Python types."""
    if isinstance(data, dict):
        return {k: make_json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_safe(v) for v in data]
    elif isinstance(data, (np.integer, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64)):
        return float(data)
    elif pd.isna(data):
        return None
    else:
        return data

# ===================================================================
# 🧠 COMPLAINT CONTEXT ANALYZER (Word2Vec + Keyword Hybrid)
# ===================================================================
class ComplaintContextAnalyzer:
    def __init__(self, complaint_csv, keyword_csv, dataset_type='training'):
        """
        complaint_csv : str  -> CSV with complaint_id, description, violation_category
        keyword_csv   : str  -> CSV with violation_type, keywords
        """
        # Load datasets
        type = f'{dataset_type}/' if dataset_type != '' else ''

        self.df = pd.read_csv(f'./dataset/{type}{complaint_csv}.csv')
        self.keyword_df = pd.read_csv(f'./dataset/{type}{keyword_csv}.csv')

        # Parse keyword strings into lists
        self.keyword_df["keywords"] = self.keyword_df["keywords"].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        )

        self.model = None
        self.violation_vectors = {}

        # Initialize model + embeddings
        self._train_word2vec()
        self._compute_violation_vectors()

        #print("✅ Context analyzer initialized successfully!")

    def _clean_word(self, word):
        """
        Normalize Tagalog and Taglish verbs:
        - 'naglaro' → 'laro'
        - 'naglalaro' → 'laro'
        - 'nagasmoking' → 'smoking'
        - 'magvape' → 'vape'
        - 'nagchat' → 'chat'
        """
        word = word.lower().strip()

        # Remove repeated letters (e.g., 'naglalaro' → 'laro')
        word = re.sub(r"(.)\1{2,}", r"\1", word)

        # Handle common prefixes (Tagword = word.lower().strip()

        # Remove repeated letters (e.g., 'naglalaro' → 'laro')
        word = re.sub(r"(.)\1{2,}", r"\1", word)

        # Handle combined Tagalog-English verbs (e.g. nagasmoking → smoking)
        prefixes = ["nag", "naga", "mag", "ma", "na", "pa", "pag", "pang"]

        for prefix in prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                rest = word[len(prefix):]

                # Remove connector vowels (e.g. "a", "e") before English stems
                if len(rest) > 1 and rest[0] in ["a", "e", "o", "u", "i"]:
                    rest = rest[1:]

                # If remainder looks like an English word, keep it
                if re.match(r"^[a-z]{3,}$", rest):
                    return rest

                # Fallback: remove common Tagalog syllables
                rest = re.sub(r"^(la|ra|na|ta|ka)+", "", rest)
                return rest

        return word



    def _tokenize(self, text):
        """
        Tokenize and clean Taglish input text.
        """
        text = text.lower()
        text = re.sub(r"[^a-záéíóúñ\s]", " ", text)
        words = text.split()

        cleaned = [self._clean_word(w) for w in words if len(w) > 2]

        stopwords = {
            "ang", "sa", "ng", "si", "ni", "kay", "and", "the", "for", "with",
            "at", "na", "nya", "sya", "ko", "ako", "ka", "ikaw", "yung", "dun",
            "dito", "doon", "yung", "kasi", "pero"
        }

        return [w for w in cleaned if w not in stopwords]
    # ----------------------------------------------------------
    # 🧠 Train Word2Vec Model
    # ----------------------------------------------------------
    def _train_word2vec(self):
        sentences = [self._tokenize(desc) for desc in self.df["description"].fillna("")]
        self.model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1, epochs=30)

    # ----------------------------------------------------------
    # 🔢 Compute Mean Embeddings per Violation Category
    # ----------------------------------------------------------
    def _compute_violation_vectors(self):
        for v in self.df["violation_category"].unique():
            tokens = []
            for desc in self.df[self.df["violation_category"] == v]["description"]:
                tokens.extend(self._tokenize(desc))
            vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
            if len(vectors) > 0:
                self.violation_vectors[v] = np.mean(vectors, axis=0)
        #(f"✅ Computed mean vectors for {len(self.violation_vectors)} violation categories.")

    # ----------------------------------------------------------
    # 🧩 Keyword-based Matching (Backup / Enhancer)
    # ----------------------------------------------------------
    def _keyword_match(self, complaint_text):
        tokens = set(self._tokenize(complaint_text))
        scores = {}

        for _, row in self.keyword_df.iterrows():
            overlap = tokens.intersection(set([k.lower() for k in row["keywords"]]))
            if overlap:
                scores[row["violation_type"]] = len(overlap) / len(row["keywords"])

        # Return top matches
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:5]

    # ----------------------------------------------------------
    # 🔍 Analyze New Complaint Text (Hybrid Approach)
    # ----------------------------------------------------------
    def analyze(self, complaint_text, top_n=10):
        tokens = self._tokenize(complaint_text)
        vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]

        # 🔹 If no vector match → fallback to keyword logic only
        if not vectors:
            keyword_ranked = self._keyword_match(complaint_text)
            if not keyword_ranked:
                return pd.DataFrame([], columns=["violation", "similarity"]).to_json(orient="records")
            df_keywords = pd.DataFrame(keyword_ranked, columns=["violation", "similarity"])
            # ✅ Filter: only push if similarity > 0.6
            df_keywords = df_keywords[df_keywords["similarity"] > 0.6]
            return df_keywords.to_json(orient="records")

        # 🔹 Compute Word2Vec similarities
        complaint_vec = np.mean(vectors, axis=0)
        scores = {
            v: float(cosine_similarity([complaint_vec], [vec])[0][0])
            for v, vec in self.violation_vectors.items()
        }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        df_ranked = pd.DataFrame(ranked, columns=["violation", "similarity"]).head(top_n)

        # 🔹 Combine with keyword scores for final hybrid ranking
        keyword_ranked = self._keyword_match(complaint_text)
        for v, kscore in keyword_ranked:
            if v in df_ranked["violation"].values:
                df_ranked.loc[df_ranked["violation"] == v, "similarity"] += kscore * 0.5  # small boost

        # ✅ Filter out low scores (< 0.6)
        df_ranked = df_ranked[df_ranked["similarity"] > 0.6]
        df_ranked["status"] = df_ranked["similarity"].apply(self._similarity_status)

        # 🔹 Sort and return top matches
        df_ranked = df_ranked.sort_values("similarity", ascending=False).reset_index(drop=True)
        return df_ranked.to_json(orient="records")
    
    def _similarity_status(self, score):
        if score >= 0.85:
            return "Very Strong Match"
        elif score >= 0.75:
            return "Strong Match"
        elif score >= 0.60:
            return "Likely Related"
        elif score >= 0.40:
            return "Possibly Related"
        else:
            return "Unclear / Needs Review"

    
    def evaluate_model(self, top_n=3, sample_size=500):
        """
        Evaluate Word2Vec-based similarity accuracy using known complaint labels.
        top_n: check if correct violation is within top N predictions.
        sample_size: number of samples to evaluate (for speed).
        """
        df_sample = self.df.sample(min(sample_size, len(self.df)), random_state=42)
        correct_top1, correct_topn = 0, 0

        for _, row in df_sample.iterrows():
            text = row["description"]
            true_violation = row["violation_category"]

            result_json = self.analyze(text, top_n=top_n)
            results = pd.read_json(result_json)

            if results.empty:
                continue

            top1_pred = results.iloc[0]["violation"]
            topn_preds = results["violation"].tolist()

            if true_violation == top1_pred:
                correct_top1 += 1
            if true_violation in topn_preds:
                correct_topn += 1

        total = len(df_sample)
        top1_acc = correct_top1 / total
        topn_acc = correct_topn / total

        print(f"📊 Word2Vec Evaluation Results ({total} samples):")
        print(f"Top-1 Accuracy: {top1_acc*100:.2f}%")
        print(f"Top-{top_n} Accuracy: {topn_acc*100:.2f}%")

        return {"top1": top1_acc, f"top{top_n}": topn_acc}
