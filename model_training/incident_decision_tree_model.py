import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score

class IncidentDecisionTree:
    def __init__(self, csv_file, logistic_model=None):
        # Load dataset
        self.df = pd.read_csv(f"./dataset/{csv_file}.csv")

        # If "Action" column doesn’t exist, generate it
        if "Action" not in self.df.columns:
            def assign_action(total_incidents):
                if total_incidents <= 10:
                    return "Warning"
                elif total_incidents <= 30:
                    return "Counseling"
                elif total_incidents <= 50:
                    return "Suspension"
                else:
                    return "Expulsion"
            self.df["Action"] = self.df["Total_Incidents"].apply(assign_action)

        # Features (X) and Target (y)
        self.X = self.df.drop(["Action"], axis=1)
        self.y = self.df["Action"]

        # Keep reference to logistic regression model
        self.logistic_model = logistic_model

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        # Build model
        self._build_model()

    def _build_model(self):
        self.model = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, zero_division=0)
        print("Accuracy:", acc)
        print("\nClassification Report:\n", report)
        return acc, report

    def show_rules(self):
        rules = export_text(self.model, feature_names=list(self.X.columns))
        print("\nDecision Tree Rules:\n", rules)
        return rules

    def predict_student(self, student_features: dict):
        # Auto-generate Will_Reoffend if logistic model is provided
        if "Will_Reoffend" not in student_features and self.logistic_model:
            lg_features = {k: v for k, v in student_features.items() if k in self.logistic_model.X.columns}
            pred_reoffend, _ = self.logistic_model.predict_student(lg_features)
            student_features["Will_Reoffend"] = pred_reoffend

        # Ensure DataFrame matches training columns
        student_df = pd.DataFrame([student_features], columns=self.X.columns)

        pred_action = self.model.predict(student_df)[0]
        return pred_action
    
"""
class IncidentDecisionTree:
    def __init__(self, csv_path):
        self.df = pd.read_csv(f"{csv_path}.csv")

        if "avg_severity" not in self.df.columns:
            self.df["avg_severity"] = (
                0.1 * self.df["minor_count"] +
                0.2 * self.df["repeated_incident_count"] +
                0.3 * self.df["major_count"]
            )
            self.df["avg_severity"] = (
                self.df["avg_severity"] - self.df["avg_severity"].min()
            ) / (self.df["avg_severity"].max() - self.df["avg_severity"].min())

        self.X = self.df.drop(columns=["will_reoffend"])
        self.y = self.df["will_reoffend"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        self.model = None

        print(f"✅ Dataset loaded with {len(self.df)} records.")
        print(f"📊 Training: {len(self.X_train)}, Testing: {len(self.X_test)}")

    def train(self, max_depth=4, criterion="entropy"):
        self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        print("\n📈 Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("AUC:", round(roc_auc_score(self.y_test, y_prob), 3))

    def predict(self, repeated_incident_count, major_count, minor_count):
        avg_severity = 0.1 * minor_count + 0.2 * repeated_incident_count + 0.3 * major_count
        avg_severity = (avg_severity - self.df["avg_severity"].min()) / (
            self.df["avg_severity"].max() - self.df["avg_severity"].min()
        )

        student = pd.DataFrame([{
            "repeated_incident_count": repeated_incident_count,
            "major_count": major_count,
            "minor_count": minor_count,
            "avg_severity": avg_severity
        }])

        pred_class = self.model.predict(student)[0]
        pred_prob = self.model.predict_proba(student)[:, 1][0]

        if pred_class == 1:
            if major_count > 3:
                remark = "⚠️ High risk: frequent major incidents suggest reoffense."
            elif repeated_incident_count > 2:
                remark = "🟠 Moderate risk: repeated patterns suggest reoffense."
            elif pred_prob > 0.7:
                remark = "⚠️ Elevated probability of reoffense."
            else:
                remark = "🟡 Mild risk of reoffense under certain conditions."
        else:
            if major_count == 0 and repeated_incident_count == 0:
                remark = "✅ Very low risk: clean record."
            elif avg_severity < 0.3:
                remark = "✅ Low risk: minor issues only."
            else:
                remark = "🟢 Stable but monitor periodically."

        print("\n🎯 Prediction Result:")
        print(f"Predicted class (Will reoffend?): {pred_class}")
        print(f"Predicted probability: {round(pred_prob, 3)}")
        print(f"📝 Remark: {remark}")

        return pred_class, pred_prob, remark
"""